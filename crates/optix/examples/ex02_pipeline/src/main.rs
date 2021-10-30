mod renderer;
use renderer::Renderer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut renderer = Renderer::new(256, 128)?;
    renderer.render()?;
    Ok(())
}