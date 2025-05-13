use std::env;
use std::error;
use std::fs;
use std::path;

/// Represents the cuDNN SDK installation.
#[derive(Debug, Clone)]
pub struct CudnnSdk {
    /// cuDNN related paths and version numbers.
    cudnn_include_path: path::PathBuf,
    cudnn_version: [u32; 3],
}

impl CudnnSdk {
    /// Creates a new `cuDNN` instance by locating the cuDNN SDK installation
    /// and parsing its version from the `cudnn_version.h` header file.
    pub fn new() -> Result<Self, Box<dyn error::Error>> {
        // Retrieve the cuDNN include paths.
        let cudnn_include_path = Self::find_cudnn_include_dir()?;
        // Retrieve the cuDNN version.
        let header_path = cudnn_include_path.join("cudnn_version.h");
        let header_content = fs::read_to_string(header_path)?;
        let cudnn_version = Self::parse_cudnn_version(header_content.as_str())?;
        Ok(Self {
            cudnn_include_path,
            cudnn_version,
        })
    }

    pub fn cudnn_include_path(&self) -> &path::Path {
        self.cudnn_include_path.as_path()
    }

    /// Returns the full version of cuDNN as an integer.
    /// For example, cuDNN 9.8.0 is represented as 90800.
    pub fn cudnn_version(&self) -> u32 {
        let [major, minor, patch] = self.cudnn_version;
        major * 10000 + minor * 100 + patch
    }

    pub fn cudnn_version_major(&self) -> u32 {
        self.cudnn_version[0]
    }

    pub fn cudnn_version_minor(&self) -> u32 {
        self.cudnn_version[1]
    }

    pub fn cudnn_version_patch(&self) -> u32 {
        self.cudnn_version[2]
    }

    /// Checks if the given path is a valid cuDNN installation by verifying
    /// the existence of cuDNN header files.
    fn is_cudnn_include_path<P: AsRef<path::Path>>(path: P) -> bool {
        let p = path.as_ref();
        p.join("cudnn.h").is_file() && p.join("cudnn_version.h").is_file()
    }

    fn find_cudnn_include_dir() -> Result<path::PathBuf, Box<dyn error::Error>> {
        #[cfg(not(target_os = "windows"))]
        const CUDNN_DEFAULT_PATHS: &[&str] = &["/usr/include", "/usr/local/include"];
        #[cfg(target_os = "windows")]
        const CUDNN_DEFAULT_PATHS: &[&str] = &[
            "C:/Program Files/NVIDIA/CUDNN/v9.x/include",
            "C:/Program Files/NVIDIA/CUDNN/v8.x/include",
        ];

        let mut cudnn_paths: Vec<String> =
            CUDNN_DEFAULT_PATHS.iter().map(|s| s.to_string()).collect();
        if let Some(override_path) = env::var_os("CUDNN_INCLUDE_DIR") {
            cudnn_paths.push(
                override_path
                    .into_string()
                    .expect("CUDNN_INCLUDE_DIR to be a Unicode string"),
            );
        }

        cudnn_paths
            .iter()
            .find(|s| Self::is_cudnn_include_path(s))
            .map(path::PathBuf::from)
            .ok_or("Cannot find cuDNN include directory.".into())
    }

    fn parse_cudnn_version(header_content: &str) -> Result<[u32; 3], Box<dyn error::Error>> {
        let [major, minor, patch] = ["CUDNN_MAJOR", "CUDNN_MINOR", "CUDNN_PATCHLEVEL"]
            .into_iter()
            .map(|macro_name| {
                let version = header_content
                    .lines()
                    .find(|line| line.contains(format!("#define {macro_name}").as_str()))
                    .and_then(|line| line.split_whitespace().last())
                    .ok_or(format!("Cannot find {macro_name} from cuDNN header file.").as_str())?;
                version
                    .parse::<u32>()
                    .map_err(|_| format!("Cannot parse {macro_name} as u32: '{}'", version))
            })
            .collect::<Result<Vec<u32>, _>>()?
            .try_into()
            .map_err(|_| "Invalid cuDNN version length.")?;
        Ok([major, minor, patch])
    }
}
