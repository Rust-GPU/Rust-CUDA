/// What to include when filtering PTX output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtxOutputMode {
    /// Include everything
    All,
    /// Include only function declarations (no bodies)
    DeclarationsOnly,
    /// Include specific functions based on filter
    Filtered,
}

/// Filter for selecting specific functions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionFilter {
    /// Include all functions
    All,
    /// Include functions with names containing this string
    ByName(String),
    /// Include only entry points with names containing this string
    EntryPoint(String),
}

/// Configuration for filtering PTX output
#[derive(Debug, Clone)]
pub struct PtxFilterConfig {
    /// What content to include
    pub mode: PtxOutputMode,

    /// Filter for selecting functions (only used when mode is Filtered)
    pub function_filter: FunctionFilter,

    /// What additional content to include
    pub include_header: bool,
    pub include_globals: bool,
}

impl Default for PtxFilterConfig {
    fn default() -> Self {
        Self {
            mode: PtxOutputMode::Filtered,
            function_filter: FunctionFilter::All,
            include_header: true,
            include_globals: true,
        }
    }
}

impl PtxFilterConfig {
    /// Create a config that includes everything
    pub fn all() -> Self {
        Self {
            mode: PtxOutputMode::All,
            function_filter: FunctionFilter::All,
            include_header: true,
            include_globals: true,
        }
    }

    /// Create a config for declarations only
    pub fn declarations_only() -> Self {
        Self {
            mode: PtxOutputMode::DeclarationsOnly,
            function_filter: FunctionFilter::All,
            include_header: true,
            include_globals: true,
        }
    }

    /// Create a config that filters by function name
    pub fn by_function_name(name: impl Into<String>) -> Self {
        Self {
            mode: PtxOutputMode::Filtered,
            function_filter: FunctionFilter::ByName(name.into()),
            include_header: false,
            include_globals: false,
        }
    }

    /// Create a config that filters by entry point name
    pub fn by_entry_point(name: impl Into<String>) -> Self {
        Self {
            mode: PtxOutputMode::Filtered,
            function_filter: FunctionFilter::EntryPoint(name.into()),
            include_header: false,
            include_globals: false,
        }
    }

    /// Create a config from CodegenArgs
    pub fn from_codegen_args(args: &crate::context::CodegenArgs) -> Self {
        use crate::context::DisassembleMode;
        match &args.disassemble {
            Some(DisassembleMode::All) => Self::all(),
            Some(DisassembleMode::Globals) => Self::declarations_only(),
            Some(DisassembleMode::Function(func_name)) => Self::by_function_name(func_name),
            Some(DisassembleMode::Entry(entry_name)) => Self::by_entry_point(entry_name),
            None => Self::default(),
        }
    }
}

/// PTX output filter that processes PTX assembly based on configuration
pub struct PtxFilter {
    config: PtxFilterConfig,
}

impl PtxFilter {
    pub fn new(config: PtxFilterConfig) -> Self {
        Self { config }
    }

    /// Filter PTX content based on the configuration
    pub fn filter(&self, ptx: &str) -> String {
        // If mode is All, return everything
        if self.config.mode == PtxOutputMode::All {
            return ptx.to_string();
        }

        let parsed = PtxContent::parse(ptx);
        parsed.format(&self.config)
    }
}

/// Parsed PTX content
#[derive(Debug, Default)]
struct PtxContent {
    header_lines: Vec<String>,
    globals: Vec<String>,
    functions: Vec<PtxFunction>,
}

/// A parsed PTX function
#[derive(Debug)]
struct PtxFunction {
    name: String,
    is_entry: bool,
    declaration_line: String,
    body_lines: Vec<String>,
}

impl PtxContent {
    /// Parse PTX text into structured content
    fn parse(ptx: &str) -> Self {
        let mut content = Self::default();
        let mut in_function = false;
        let mut current_function: Option<PtxFunction> = None;

        for line in ptx.lines() {
            if Self::is_header_line(line) {
                content.header_lines.push(line.to_string());
            } else if Self::is_global_line(line) && !in_function {
                content.globals.push(line.to_string());
            } else if let Some(func) = Self::parse_function_start(line) {
                // Save previous function if any
                if let Some(f) = current_function.take() {
                    content.functions.push(f);
                }
                current_function = Some(func);
                in_function = true;
            } else if in_function && let Some(ref mut func) = current_function {
                func.body_lines.push(line.to_string());
                if line.trim() == "}" {
                    content.functions.push(current_function.take().unwrap());
                    in_function = false;
                }
            }
        }

        // Handle case where file ends while in function
        if let Some(func) = current_function {
            content.functions.push(func);
        }

        content
    }

    fn is_header_line(line: &str) -> bool {
        line.starts_with(".version")
            || line.starts_with(".target")
            || line.starts_with(".address_size")
    }

    fn is_global_line(line: &str) -> bool {
        line.contains(".global") || line.contains(".const") || line.contains(".shared")
    }

    fn parse_function_start(line: &str) -> Option<PtxFunction> {
        if line.contains(".func") || line.contains(".entry") {
            Some(PtxFunction {
                name: Self::extract_function_name(line),
                is_entry: line.contains(".entry"),
                declaration_line: line.to_string(),
                body_lines: vec![],
            })
        } else {
            None
        }
    }

    fn extract_function_name(line: &str) -> String {
        // Look for patterns like:
        // .visible .entry kernel_main(
        // .func (.reg .u32 %ret) helper_func()
        // .entry simple_kernel (

        // Strategy: Find all potential function names (valid identifiers)
        // The last one before the final '(' is usually the function name
        let mut potential_names = Vec::new();
        let mut current_word = String::new();
        let mut paren_depth = 0;

        for ch in line.chars() {
            match ch {
                '(' => {
                    // Save any current word before we see a paren
                    if !current_word.is_empty()
                        && current_word
                            .chars()
                            .all(|c| c.is_alphanumeric() || c == '_')
                        && paren_depth == 0
                    {
                        potential_names.push(current_word.clone());
                    }
                    current_word.clear();
                    paren_depth += 1;
                }
                ')' => {
                    current_word.clear();
                    if paren_depth > 0 {
                        paren_depth -= 1;
                    }
                }
                ' ' | '\t' | ',' | '.' => {
                    if !current_word.is_empty()
                        && current_word
                            .chars()
                            .all(|c| c.is_alphanumeric() || c == '_')
                        && paren_depth == 0
                    {
                        // This is a word at depth 0 (not inside parentheses)
                        potential_names.push(current_word.clone());
                    }
                    current_word.clear();
                }
                _ => {
                    if ch.is_alphanumeric() || ch == '_' {
                        current_word.push(ch);
                    } else {
                        current_word.clear();
                    }
                }
            }
        }

        // Handle case where line ends with the function name
        if !current_word.is_empty()
            && current_word
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_')
        {
            potential_names.push(current_word);
        }

        // Return the last potential name found, or empty string
        potential_names.into_iter().last().unwrap_or_default()
    }

    /// Format the parsed content according to the configuration
    fn format(&self, config: &PtxFilterConfig) -> String {
        let mut output = String::new();

        // Add header if requested
        if config.include_header {
            for line in &self.header_lines {
                output.push_str(line);
                output.push('\n');
            }
        }

        // Add globals if requested
        if config.include_globals {
            for line in &self.globals {
                output.push_str(line);
                output.push('\n');
            }
        }

        // Add functions based on mode
        match config.mode {
            PtxOutputMode::All => {
                // Already handled above
                unreachable!()
            }
            PtxOutputMode::DeclarationsOnly => {
                for func in &self.functions {
                    output.push_str(&func.declaration_line);
                    output.push_str(" { ... }\n\n");
                }
            }
            PtxOutputMode::Filtered => {
                for func in &self.functions {
                    if self.should_include_function(func, &config.function_filter) {
                        output.push_str(&func.declaration_line);
                        output.push('\n');
                        for line in &func.body_lines {
                            output.push_str(line);
                            output.push('\n');
                        }
                        output.push('\n');
                    }
                }
            }
        }

        output
    }

    fn should_include_function(&self, func: &PtxFunction, filter: &FunctionFilter) -> bool {
        match filter {
            FunctionFilter::All => true,
            FunctionFilter::ByName(name) => func.name.contains(name),
            FunctionFilter::EntryPoint(name) => func.is_entry && func.name.contains(name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PTX: &str = r#".version 8.7
.target sm_61, debug
.address_size 64

.global .align 4 .u32 global_var = 42;

.visible .entry kernel_main(
    .param .u64 kernel_main_param_0
)
{
    .reg .u64 %r1;
    ld.param.u64 %r1, [kernel_main_param_0];
    ret;
}

.func (.reg .u32 %ret) helper_func()
{
    .reg .u32 %r1;
    mov.u32 %r1, 10;
    mov.u32 %ret, %r1;
    ret;
}

.visible .entry another_kernel()
{
    ret;
}
"#;

    #[test]
    fn test_filter_all() {
        let config = PtxFilterConfig::all();
        let filter = PtxFilter::new(config);
        let result = filter.filter(SAMPLE_PTX);
        assert_eq!(result, SAMPLE_PTX);
    }

    #[test]
    fn test_filter_by_entry_point() {
        let config = PtxFilterConfig::by_entry_point("kernel_main");
        let filter = PtxFilter::new(config);
        let result = filter.filter(SAMPLE_PTX);

        // Should NOT include header or globals with new config
        assert!(!result.contains(".version 8.7"));
        assert!(!result.contains(".target sm_61"));
        assert!(!result.contains(".address_size 64"));
        assert!(!result.contains(".global .align 4 .u32 global_var"));

        // Should include kernel_main
        assert!(result.contains(".visible .entry kernel_main"));
        assert!(result.contains("ld.param.u64 %r1"));

        // Should NOT include helper_func or another_kernel
        assert!(!result.contains("helper_func"));
        assert!(!result.contains("another_kernel"));
    }

    #[test]
    fn test_filter_by_function_name() {
        let config = PtxFilterConfig::by_function_name("helper_func");
        let filter = PtxFilter::new(config);
        let result = filter.filter(SAMPLE_PTX);

        // Should NOT include header with new config
        assert!(!result.contains(".version 8.7"));

        // Should include helper_func
        assert!(result.contains(".func (.reg .u32 %ret) helper_func"));
        assert!(result.contains("mov.u32 %r1, 10"));

        // Should NOT include kernels
        assert!(!result.contains("kernel_main"));
        assert!(!result.contains("another_kernel"));
    }

    #[test]
    fn test_declarations_only() {
        let config = PtxFilterConfig::declarations_only();
        let filter = PtxFilter::new(config);
        let result = filter.filter(SAMPLE_PTX);

        // Should include header
        assert!(result.contains(".version 8.7"));

        // Should include globals
        assert!(result.contains(".global .align 4 .u32 global_var"));

        // Should include function declarations but not bodies
        assert!(result.contains(".visible .entry kernel_main"));
        assert!(result.contains(" { ... }"));
        assert!(!result.contains("ld.param.u64"));
    }

    #[test]
    fn test_partial_name_match() {
        let config = PtxFilterConfig::by_entry_point("kernel");
        let filter = PtxFilter::new(config);
        let result = filter.filter(SAMPLE_PTX);

        // Should include both kernels that contain "kernel"
        assert!(result.contains("kernel_main"));
        assert!(result.contains("another_kernel"));

        // Should NOT include helper_func
        assert!(!result.contains("helper_func"));
    }

    #[test]
    fn test_extract_function_name() {
        assert_eq!(
            PtxContent::extract_function_name(".visible .entry kernel_main("),
            "kernel_main"
        );
        assert_eq!(
            PtxContent::extract_function_name(".func (.reg .u32 %ret) helper_func()"),
            "helper_func"
        );
        assert_eq!(
            PtxContent::extract_function_name(".entry simple_kernel ("),
            "simple_kernel"
        );
    }
}
