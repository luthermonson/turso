//! Schema dialects.
//!
//! The [`SchemaDialect`] trait is the boundary between the engine and a
//! frontend's SQL surface. The engine owns schema *mechanics* (root pages,
//! B-trees, the `sqlite_schema` table itself) and goes through the dialect
//! for schema *text*: whenever it parses a `sqlite_schema` table row back
//! into a table definition, the dialect decides how that stored SQL is
//! interpreted. The [`sqlite`] module owns the default dialect and the
//! catalog tables that ship with every Turso build (`pragma_*`,
//! `json_each`/`json_tree`, `sqlite_dbpage`, `btree_dump`,
//! `sqlite_turso_types`).

pub mod sqlite;

pub use sqlite::SqliteSchemaDialect;

/// Schema dialect layered on top of the engine.
///
/// Every [`crate::Database`] carries a dialect, fixed at open time and
/// [`SqliteSchemaDialect`] by default. The engine consults it on every
/// schema load — database open, connection schema reparse, the
/// `ParseSchema` opcode after DDL, and MVCC schema rebuild — so a frontend
/// can store its own schema SQL in `sqlite_schema` and reparse it here.
pub trait SchemaDialect: Send + Sync + 'static {
    /// Stable identifier for this dialect (e.g. "sqlite", "postgres").
    ///
    /// A database file must always be opened with the same dialect it was
    /// created with; the process-wide database registry uses this name to
    /// reject an open whose dialect differs from the already-open instance.
    fn name(&self) -> &'static str;

    /// Parse a `sqlite_schema` `type='table'` row's SQL into a table
    /// definition.
    ///
    /// Rows written by internal engine paths (sequence backing tables,
    /// `sqlite_sequence`) are plain SQLite text and carry no frontend
    /// marker, so every implementation must fall back to SQLite parsing
    /// for text it does not recognize as its own.
    fn parse_table_sql(
        &self,
        sql: &str,
        root_page: i64,
    ) -> crate::Result<crate::schema::BTreeTable>;

    /// Produce the SQL text to store in `sqlite_schema` for a
    /// `CREATE TABLE`.
    ///
    /// `input` is the original statement text as the user wrote it, in the
    /// frontend's dialect; `tbl_name` and `body` are the translated AST.
    /// The SQLite dialect formats canonical SQLite text from the AST; a
    /// frontend dialect typically stores `input` with a marker it can
    /// recognize in [`SchemaDialect::parse_table_sql`].
    fn format_table_sql(
        &self,
        input: &str,
        tbl_name: &turso_parser::ast::QualifiedName,
        body: &turso_parser::ast::CreateTableBody,
    ) -> crate::Result<String>;

    /// Install the dialect's catalog tables into a freshly constructed
    /// schema.
    ///
    /// Called by [`crate::schema::Schema::with_options`] on every schema
    /// construction and rebuild, so catalog tables survive rebuilds
    /// structurally instead of being re-registered by hand. The SQLite
    /// dialect registers the standard built-in catalog here; other
    /// dialects typically compose with it via
    /// [`sqlite::register_builtin_catalog`] and then add their own tables
    /// (constructed with [`crate::VirtualTable::new_internal`], which
    /// requires no connection).
    fn register_catalog(
        &self,
        schema: &mut crate::schema::Schema,
        enable_custom_types: bool,
    ) -> crate::Result<()>;

    /// Resolve a function name in user SQL to the engine's function IR.
    ///
    /// The dialect owns its scalar function surface: the SQLite dialect
    /// resolves the built-in set, another dialect resolves its own —
    /// mapping names onto engine primitives where it wants them (usually
    /// by composing with [`sqlite::resolve_builtin_function`]) and onto
    /// [`crate::Func::Dialect`] for functions it executes
    /// itself via [`SchemaDialect::exec_scalar_function`]. Consulted
    /// before extension functions; engine-generated helper statements
    /// always resolve with SQLite semantics instead.
    fn resolve_function(
        &self,
        name: &str,
        arg_count: usize,
    ) -> crate::Result<Option<crate::Func>>;

    /// Execute a dialect scalar function at runtime.
    ///
    /// Receives the connection — unlike extension functions — because
    /// catalog functions (e.g. `pg_get_tabledef`) need to inspect the
    /// schema. Only reached through [`crate::Func::Dialect`],
    /// so a dialect that never resolves to that variant can keep the
    /// default "no such function" error.
    fn exec_scalar_function(
        &self,
        _conn: &crate::Connection,
        name: &str,
        _args: &[crate::Value],
    ) -> crate::Result<crate::Value> {
        Err(crate::LimboError::ParseError(format!(
            "no such function: {name}"
        )))
    }

    /// Whether this dialect needs the custom-type machinery (DECODE/ENCODE,
    /// affinity metadata) regardless of the experimental database flag.
    /// A dialect whose type system leans on custom types (e.g. PostgreSQL)
    /// returns true so its databases never open with the machinery off.
    fn requires_custom_types(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::BTreeTable;
    use crate::storage::database::DatabaseFile;
    use crate::sync::atomic::{AtomicUsize, Ordering};
    use crate::{Database, DatabaseOpts, MemoryIO, OpenFlags, IO};
    use std::sync::Arc;

    /// A dialect that counts schema-row parses and strips a `/* test */ `
    /// marker before delegating to SQLite parsing, mirroring how a frontend
    /// dialect recognizes its own stored text and falls back to SQLite for
    /// unmarked rows.
    #[derive(Default)]
    struct TestSchemaDialect {
        parse_calls: AtomicUsize,
    }

    impl SchemaDialect for TestSchemaDialect {
        fn name(&self) -> &'static str {
            "test"
        }

        fn parse_table_sql(&self, sql: &str, root_page: i64) -> crate::Result<BTreeTable> {
            self.parse_calls.fetch_add(1, Ordering::SeqCst);
            let sql = sql.strip_prefix("/* test */ ").unwrap_or(sql);
            BTreeTable::from_sql(sql, root_page)
        }

        fn format_table_sql(
            &self,
            input: &str,
            _tbl_name: &turso_parser::ast::QualifiedName,
            _body: &turso_parser::ast::CreateTableBody,
        ) -> crate::Result<String> {
            Ok(format!("/* test */ {input}"))
        }

        fn resolve_function(
            &self,
            name: &str,
            arg_count: usize,
        ) -> crate::Result<Option<crate::function::Func>> {
            if name.eq_ignore_ascii_case("test_add_one") && arg_count == 1 {
                return Ok(Some(crate::function::Func::Dialect(
                    "test_add_one".to_string(),
                )));
            }
            sqlite::resolve_builtin_function(name, arg_count)
        }

        fn exec_scalar_function(
            &self,
            _conn: &crate::Connection,
            name: &str,
            args: &[crate::Value],
        ) -> crate::Result<crate::Value> {
            assert_eq!(name, "test_add_one");
            let crate::Value::Numeric(crate::numeric::Numeric::Integer(v)) = args[0] else {
                return Err(crate::LimboError::InvalidArgument(
                    "test_add_one expects an integer".to_string(),
                ));
            };
            Ok(crate::Value::Numeric(crate::numeric::Numeric::Integer(
                v + 1,
            )))
        }

        fn requires_custom_types(&self) -> bool {
            true
        }

        fn register_catalog(
            &self,
            schema: &mut crate::schema::Schema,
            enable_custom_types: bool,
        ) -> crate::Result<()> {
            sqlite::register_builtin_catalog(schema, enable_custom_types)?;
            let vtab = crate::VirtualTable::new_internal(
                "test_catalog".to_string(),
                "CREATE TABLE test_catalog (value INTEGER)".to_string(),
                turso_ext::VTabKind::VirtualTable,
                Arc::new(crate::sync::RwLock::new(TestCatalogTable)),
            )?;
            schema.add_virtual_table(Arc::new(vtab))
        }
    }

    /// A one-row catalog table installed by [`TestSchemaDialect`],
    /// standing in for a frontend catalog surface like `pg_class`.
    #[derive(Debug)]
    struct TestCatalogTable;

    impl crate::InternalVirtualTable for TestCatalogTable {
        fn name(&self) -> String {
            "test_catalog".to_string()
        }

        fn sql(&self) -> String {
            "CREATE TABLE test_catalog (value INTEGER)".to_string()
        }

        fn open(
            &self,
            _conn: Arc<crate::Connection>,
        ) -> crate::Result<Arc<crate::sync::RwLock<dyn crate::InternalVirtualTableCursor>>>
        {
            Ok(Arc::new(crate::sync::RwLock::new(TestCatalogCursor {
                row: 0,
            })))
        }

        fn best_index(
            &self,
            constraints: &[turso_ext::ConstraintInfo],
            _order_by: &[turso_ext::OrderByInfo],
        ) -> std::result::Result<turso_ext::IndexInfo, turso_ext::ResultCode> {
            Ok(turso_ext::IndexInfo {
                idx_num: 0,
                idx_str: None,
                order_by_consumed: false,
                estimated_cost: 1.0,
                estimated_rows: 1,
                constraint_usages: constraints
                    .iter()
                    .map(|_| turso_ext::ConstraintUsage {
                        argv_index: None,
                        omit: false,
                    })
                    .collect(),
            })
        }
    }

    struct TestCatalogCursor {
        row: usize,
    }

    impl crate::InternalVirtualTableCursor for TestCatalogCursor {
        fn filter(
            &mut self,
            _args: &[crate::Value],
            _idx_str: Option<String>,
            _idx_num: i32,
        ) -> crate::Result<bool> {
            self.row = 0;
            Ok(true)
        }

        fn next(&mut self) -> crate::Result<bool> {
            self.row += 1;
            Ok(self.row < 1)
        }

        fn rowid(&self) -> i64 {
            self.row as i64
        }

        fn column(&self, column: usize) -> crate::Result<crate::Value> {
            match column {
                0 => Ok(crate::Value::Numeric(crate::numeric::Numeric::Integer(42))),
                _ => Ok(crate::Value::Null),
            }
        }
    }

    fn open_db(
        io: &Arc<dyn IO>,
        path: &str,
        dialect: Arc<dyn SchemaDialect>,
    ) -> crate::Result<Arc<Database>> {
        let file = io.open_file(path, OpenFlags::Create, true)?;
        let db_file = Arc::new(DatabaseFile::new(file));
        Database::open_with_flags_with_allocator(
            io.clone(),
            path,
            db_file,
            OpenFlags::default(),
            DatabaseOpts::new(),
            None,
            None,
            crate::alloc::DynAllocator::default(),
            dialect,
        )
    }

    #[test]
    fn schema_load_routes_through_dialect() {
        let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
        {
            let db = open_db(&io, "dialect-load.db", Arc::new(SqliteSchemaDialect)).unwrap();
            let conn = db.connect().unwrap();
            conn.execute("CREATE TABLE t (x INTEGER)").unwrap();
            conn.close().unwrap();
        }

        // Reopening the database parses the stored schema row for `t`
        // through the dialect.
        let dialect = Arc::new(TestSchemaDialect::default());
        let db = open_db(&io, "dialect-load.db", dialect.clone()).unwrap();
        assert!(dialect.parse_calls.load(Ordering::SeqCst) >= 1);

        // DDL reparses the schema via the ParseSchema opcode, again through
        // the dialect.
        let conn = db.connect().unwrap();
        let before = dialect.parse_calls.load(Ordering::SeqCst);
        conn.execute("CREATE TABLE u (y INTEGER)").unwrap();
        assert!(dialect.parse_calls.load(Ordering::SeqCst) > before);

        // Both tables are usable under the dialect.
        conn.execute("INSERT INTO t VALUES (1)").unwrap();
        conn.execute("INSERT INTO u VALUES (2)").unwrap();
        conn.close().unwrap();
    }

    #[test]
    fn dialect_catalog_available_on_every_schema_and_rebuild() {
        let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
        let db = open_db(
            &io,
            "dialect-catalog.db",
            Arc::new(TestSchemaDialect::default()),
        )
        .unwrap();

        let query_catalog = |conn: &Arc<crate::Connection>| -> Vec<Vec<crate::Value>> {
            conn.prepare("SELECT value FROM test_catalog")
                .unwrap()
                .run_collect_rows()
                .unwrap()
        };

        let conn1 = db.connect().unwrap();
        let conn2 = db.connect().unwrap();
        assert_eq!(
            query_catalog(&conn1),
            vec![vec![crate::Value::Numeric(
                crate::numeric::Numeric::Integer(42)
            )]]
        );
        assert_eq!(
            query_catalog(&conn2),
            vec![vec![crate::Value::Numeric(
                crate::numeric::Numeric::Integer(42)
            )]]
        );

        // DDL on another connection forces conn1 to rebuild its schema from
        // sqlite_schema; the catalog table must survive because schema
        // construction re-registers it.
        conn2.execute("CREATE TABLE t (x INTEGER)").unwrap();
        assert_eq!(
            query_catalog(&conn1),
            vec![vec![crate::Value::Numeric(
                crate::numeric::Numeric::Integer(42)
            )]]
        );

        conn1.close().unwrap();
        conn2.close().unwrap();
    }

    #[test]
    fn create_table_stores_dialect_formatted_sql() {
        let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
        {
            let dialect = Arc::new(TestSchemaDialect::default());
            let db = open_db(&io, "dialect-store.db", dialect).unwrap();
            let conn = db.connect().unwrap();

            // A frontend prepares its translated AST while supplying the
            // original statement text.
            let input = "CREATE TABLE t (x INTEGER)";
            let stmt = match turso_parser::parser::Parser::new(input.as_bytes())
                .next_cmd()
                .unwrap()
                .unwrap()
            {
                turso_parser::ast::Cmd::Stmt(stmt) => stmt,
                other => panic!("unexpected command: {other:?}"),
            };
            conn.prepare_translated_stmt(stmt, input)
                .unwrap()
                .run_ignore_rows()
                .unwrap();

            // The stored schema row carries the dialect marker and the
            // original text.
            let rows = conn
                .prepare("SELECT sql FROM sqlite_schema WHERE name = 't'")
                .unwrap()
                .run_collect_rows()
                .unwrap();
            assert_eq!(rows.len(), 1);
            let stored = rows[0][0].to_string();
            assert_eq!(stored.trim_matches('\''), format!("/* test */ {input}"));
            conn.close().unwrap();
        }

        // Round-trip: reopening parses the marked row back through the
        // dialect and the table stays usable.
        let dialect = Arc::new(TestSchemaDialect::default());
        let db = open_db(&io, "dialect-store.db", dialect.clone()).unwrap();
        assert!(dialect.parse_calls.load(Ordering::SeqCst) >= 1);
        let conn = db.connect().unwrap();
        conn.execute("INSERT INTO t VALUES (1)").unwrap();
        let rows = conn
            .prepare("SELECT x FROM t")
            .unwrap()
            .run_collect_rows()
            .unwrap();
        assert_eq!(rows.len(), 1);
        conn.close().unwrap();
    }

    /// A dialect that resolves no functions at all, proving the dialect —
    /// not the engine — owns the function name surface of user SQL.
    struct NoFunctionsDialect;

    impl SchemaDialect for NoFunctionsDialect {
        fn name(&self) -> &'static str {
            "nofuncs"
        }

        fn parse_table_sql(&self, sql: &str, root_page: i64) -> crate::Result<BTreeTable> {
            BTreeTable::from_sql(sql, root_page)
        }

        fn format_table_sql(
            &self,
            _input: &str,
            tbl_name: &turso_parser::ast::QualifiedName,
            body: &turso_parser::ast::CreateTableBody,
        ) -> crate::Result<String> {
            Ok(format!(
                "CREATE TABLE {} {}",
                tbl_name.name.as_ident(),
                body
            ))
        }

        fn register_catalog(
            &self,
            schema: &mut crate::schema::Schema,
            enable_custom_types: bool,
        ) -> crate::Result<()> {
            sqlite::register_builtin_catalog(schema, enable_custom_types)
        }

        fn resolve_function(
            &self,
            _name: &str,
            _arg_count: usize,
        ) -> crate::Result<Option<crate::function::Func>> {
            Ok(None)
        }
    }

    #[test]
    fn dialect_scalar_function_resolves_and_executes() {
        let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
        let db = open_db(
            &io,
            "dialect-funcs.db",
            Arc::new(TestSchemaDialect::default()),
        )
        .unwrap();
        let conn = db.connect().unwrap();

        // Dialect-provided scalar executes through exec_scalar_function.
        let rows = conn
            .prepare("SELECT test_add_one(41)")
            .unwrap()
            .run_collect_rows()
            .unwrap();
        assert_eq!(
            rows,
            vec![vec![crate::Value::Numeric(
                crate::numeric::Numeric::Integer(42)
            )]]
        );

        // Built-ins still resolve because the dialect composes with the
        // shared table.
        let rows = conn
            .prepare("SELECT abs(-7)")
            .unwrap()
            .run_collect_rows()
            .unwrap();
        assert_eq!(
            rows,
            vec![vec![crate::Value::Numeric(
                crate::numeric::Numeric::Integer(7)
            )]]
        );

        // Unknown names still error.
        let err = conn.prepare("SELECT no_such_function(1)").unwrap_err();
        assert!(err.to_string().contains("no such function"));
        conn.close().unwrap();
    }

    #[test]
    fn dialect_owns_the_function_surface() {
        let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
        let db = open_db(&io, "dialect-nofuncs.db", Arc::new(NoFunctionsDialect)).unwrap();
        let conn = db.connect().unwrap();

        // Function-free SQL works, including DDL (whose internal helper
        // statements resolve with SQLite semantics regardless of dialect).
        conn.execute("CREATE TABLE t (x INTEGER)").unwrap();
        conn.execute("INSERT INTO t VALUES (-7)").unwrap();

        // A SQLite built-in is not part of this dialect's surface.
        let err = conn.prepare("SELECT abs(x) FROM t").unwrap_err();
        assert!(
            err.to_string().contains("no such function"),
            "unexpected error: {err}"
        );
        conn.close().unwrap();
    }

    #[test]
    fn dialect_can_require_custom_types() {
        let io: Arc<dyn IO> = Arc::new(MemoryIO::new());

        // The default dialect still gates custom types on the experimental
        // flag.
        let db = open_db(
            &io,
            "dialect-types-default.db",
            Arc::new(SqliteSchemaDialect),
        )
        .unwrap();
        let conn = db.connect().unwrap();
        let err = conn
            .execute("CREATE TYPE mytype(value text) BASE text")
            .unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("type"),
            "unexpected error: {err}"
        );
        conn.close().unwrap();

        // A dialect that requires custom types gets the machinery without
        // the flag.
        let db = open_db(
            &io,
            "dialect-types-required.db",
            Arc::new(TestSchemaDialect::default()),
        )
        .unwrap();
        let conn = db.connect().unwrap();
        conn.execute("CREATE TYPE mytype(value text) BASE text")
            .unwrap();
        conn.execute("CREATE TABLE t (x mytype)").unwrap();
        conn.execute("INSERT INTO t VALUES ('hello')").unwrap();
        let rows = conn
            .prepare("SELECT x FROM t")
            .unwrap()
            .run_collect_rows()
            .unwrap();
        assert_eq!(rows.len(), 1);
        conn.close().unwrap();
    }

    #[test]
    fn registry_rejects_dialect_mismatch() {
        let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
        let _db = open_db(&io, "dialect-mismatch.db", Arc::new(SqliteSchemaDialect)).unwrap();

        let err = open_db(
            &io,
            "dialect-mismatch.db",
            Arc::new(TestSchemaDialect::default()),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("schema dialect"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn registry_rejects_default_open_of_dialect_database() {
        let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
        let _db = open_db(
            &io,
            "dialect-mismatch-reverse.db",
            Arc::new(TestSchemaDialect::default()),
        )
        .unwrap();

        let err = open_db(
            &io,
            "dialect-mismatch-reverse.db",
            Arc::new(SqliteSchemaDialect),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("schema dialect"),
            "unexpected error: {err}"
        );
    }
}
