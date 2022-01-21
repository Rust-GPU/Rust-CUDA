#![allow(warnings)]

mod renderer;
use renderer::Renderer;

use glam::{vec4, Vec4};

mod gl_util;
use gl_util::FullscreenQuad;
use glfw::{Action, Context, Key};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(4, 1));
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));

    let mut width = 960u32;
    let mut height = 540u32;

    let mut renderer = Renderer::new(width, height)?;

    let (mut window, events) = glfw
        .create_window(
            width,
            height,
            "Example 04: mesh",
            glfw::WindowMode::Windowed,
        )
        .expect("failed to create glfw window");

    window.set_key_polling(true);
    window.make_current();

    // retina displays will return a higher res for the framebuffer
    // which we need to use for the viewport
    let (fb_width, fb_height) = window.get_framebuffer_size();

    gl::load_with(|s| glfw.get_proc_address_raw(s) as *const std::os::raw::c_void);

    let mut fsq = FullscreenQuad::new(width, height).unwrap();

    let mut image_data = vec![vec4(0.0, 0.0, 0.0, 0.0); (width * height) as usize];

    unsafe {
        gl::Viewport(0, 0, fb_width, fb_height);
    };

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }

        let (w, h) = window.get_framebuffer_size();
        let w = w as u32;
        let h = h as u32;
        if w != width || h != height {
            fsq.resize(w, h);
            renderer.resize(w, h)?;
            width = w;
            height = h;
            image_data.resize((width * height) as usize, vec4(0.0, 0.0, 0.0, 0.0));
        }

        renderer.render()?;
        renderer.download_pixels(&mut image_data)?;
        fsq.update_texture(&image_data);
        fsq.set_progression(1);

        // draw the quad
        fsq.draw();

        window.swap_buffers();
    }

    renderer.render()?;
    Ok(())
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => window.set_should_close(true),
        _ => {}
    }
}
