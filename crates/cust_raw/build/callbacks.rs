use std::cell;
use std::fs;
use std::path;
use std::sync;

use bimap;
use bindgen::callbacks::{ItemInfo, ItemKind, MacroParsingBehavior, ParseCallbacks};

/// Struct to handle renaming of functions through macro expansion.
#[derive(Debug)]
pub(crate) struct FunctionRenames {
    func_prefix: &'static str,
    out_dir: path::PathBuf,
    includes: path::PathBuf,
    include_dirs: Vec<path::PathBuf>,
    macro_names: cell::RefCell<Vec<String>>,
    func_remaps: sync::OnceLock<bimap::BiHashMap<String, String>>,
}

impl FunctionRenames {
    pub fn new<P: AsRef<path::Path>, I: Into<path::PathBuf>>(
        func_prefix: &'static str,
        out_dir: P,
        includes: I,
        include_dirs: Vec<path::PathBuf>,
    ) -> Self {
        Self {
            func_prefix,
            out_dir: out_dir.as_ref().to_path_buf(),
            includes: includes.into(),
            include_dirs,
            macro_names: cell::RefCell::new(Vec::new()),
            func_remaps: sync::OnceLock::new(),
        }
    }

    fn record_macro(&self, name: &str) {
        self.macro_names.borrow_mut().push(name.to_string());
    }

    fn expand(&self) -> &bimap::BiHashMap<String, String> {
        self.func_remaps.get_or_init(|| {
            let expand_me = self.out_dir.join("expand_macros.c");
            let includes = fs::read_to_string(&self.includes)
                .expect("Failed to read includes for function renames");

            let mut template = format!(
                r#"{includes}
#define RENAMED2(from, to) RUST_RENAMED##from##_TO_##to
#define RENAMED(from, to) RENAMED2(from, to)
"#
            );

            for name in self.macro_names.borrow().iter() {
                template.push_str(&format!("RENAMED(_{name}, {name})\n"));
            }

            {
                let mut temp = fs::File::create(&expand_me).unwrap();
                std::io::Write::write_all(&mut temp, template.as_bytes()).unwrap();
            }

            let mut build = cc::Build::new();
            build
                .file(&expand_me)
                .includes(&self.include_dirs)
                .cargo_warnings(false);

            let expanded = match build.try_expand() {
                Ok(expanded) => expanded,
                Err(e) => panic!("Failed to expand macros: {}", e),
            };
            let expanded = str::from_utf8(&expanded).unwrap();

            let mut remaps = bimap::BiHashMap::new();
            for line in expanded.lines().rev() {
                let rename_prefix = "RUST_RENAMED_";

                if let Some((original, expanded)) = line
                    .strip_prefix(rename_prefix)
                    .and_then(|s| s.split_once("_TO_"))
                    .filter(|(l, r)| l != r && !r.is_empty())
                {
                    remaps.insert(original.to_string(), expanded.to_string());
                }
            }

            fs::remove_file(&expand_me).expect("Failed to remove temporary file");
            remaps
        })
    }
}

impl ParseCallbacks for FunctionRenames {
    fn will_parse_macro(&self, name: &str) -> MacroParsingBehavior {
        if name.starts_with(self.func_prefix) {
            self.record_macro(name);
        }
        MacroParsingBehavior::Default
    }

    fn generated_name_override(&self, item_info: ItemInfo<'_>) -> Option<String> {
        let remaps = self.expand();
        match item_info.kind {
            ItemKind::Function => remaps.get_by_right(item_info.name).cloned(),
            _ => None,
        }
    }

    fn generated_link_name_override(&self, item_info: ItemInfo<'_>) -> Option<String> {
        let remaps = self.expand();
        match item_info.kind {
            ItemKind::Function => remaps.get_by_left(item_info.name).cloned(),
            _ => None,
        }
    }
}
