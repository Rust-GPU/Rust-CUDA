use std::env;
use std::error;
use std::fs;
use std::path;

/// Represents the OptiX SDK installation.
#[derive(Debug, Clone)]
pub struct OptiXSdk {
    /// The root directory of the OptiX SDK installation.
    optix_root: path::PathBuf,
    optix_include_paths: Vec<path::PathBuf>,
    /// The version of the OptiX SDK, represented as an integer (e.g., 90000 for OptiX 9.0.0).
    optix_version: u32,
}

impl OptiXSdk {
    /// Creates a new `OptiXSdk` instance by locating the OptiX SDK installation
    /// and parsing its version from the `optix.h` header file.
    ///
    /// # Errors
    /// Returns an error if the OptiX SDK cannot be found, if the version cannot be parsed
    /// or cust_raw package does not provide metadata information.
    pub fn new() -> Result<Self, Box<dyn error::Error>> {
        let optix_root = Self::find_optix_root().ok_or("OptiX SDK cannot be found.")?;
        // Retrieve the OptiX VERSION.
        let header_path = optix_root.join("include").join("optix.h");
        let header_content = fs::read_to_string(header_path)?;
        let optix_version = Self::parse_optix_version(header_content.as_str())?;
        // Retrieve the OptiX include paths.
        let optix_include_paths = vec![optix_root.join("include")];

        Ok(Self {
            optix_root,
            optix_include_paths,
            optix_version,
        })
    }

    pub fn optix_root(&self) -> &path::Path {
        &self.optix_root
    }

    pub fn optix_include_paths(&self) -> &[path::PathBuf] {
        &self.optix_include_paths
    }

    pub fn optix_version(&self) -> u32 {
        self.optix_version
    }

    pub fn optix_version_major(&self) -> u32 {
        self.optix_version / 10000
    }

    pub fn optix_version_minor(&self) -> u32 {
        (self.optix_version % 10000) / 100
    }

    pub fn optix_version_micro(&self) -> u32 {
        self.optix_version % 100
    }

    fn find_optix_root() -> Option<path::PathBuf> {
        // the optix SDK installer sets OPTIX_ROOT_DIR whenever it installs.
        // We also check OPTIX_ROOT first in case someone wants to override it without overriding
        // the SDK-set variable.
        env::var("OPTIX_ROOT")
            .ok()
            .or_else(|| env::var("OPTIX_ROOT_DIR").ok())
            .map(path::PathBuf::from)
    }

    /// Parses the content of the `optix.h` header file to extract the OptiX version.
    ///
    /// # Errors
    /// Returns an error if the `OPTIX_VERSION` definition cannot be found or parsed.
    fn parse_optix_version(header_content: &str) -> Result<u32, Box<dyn error::Error>> {
        let version = header_content
            .lines()
            .find(|line| line.contains("#define OPTIX_VERSION"))
            .and_then(|line| line.split_whitespace().last())
            .ok_or("Cannot find OPTIX_VERSION from OptiX header file.")?;
        let version = version
            .parse::<u32>()
            .map_err(|_| format!("Cannot parse OPTIX_VERSION as u32: '{}'", version))?;
        Ok(version)
    }
}
