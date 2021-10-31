use gl;
use gl::types::{GLchar, GLenum, GLint, GLsizeiptr, GLuint, GLvoid};
use std::ffi::{CStr, CString};

use crate::vector::*;

pub struct Shader {
    id: GLuint,
}

impl Shader {
    pub fn from_source(
        source: &CStr,
        shader_type: GLenum,
    ) -> Result<Shader, String> {
        let id = unsafe { gl::CreateShader(shader_type) };

        unsafe {
            gl::ShaderSource(id, 1, &source.as_ptr(), std::ptr::null());
            gl::CompileShader(id);
        }

        let mut success: GLint = 1;
        unsafe {
            gl::GetShaderiv(id, gl::COMPILE_STATUS, &mut success);
        }

        if success == 0 {
            let mut len: GLint = 0;
            unsafe {
                gl::GetShaderiv(id, gl::INFO_LOG_LENGTH, &mut len);
            }
            let error = create_whitespace_cstring(len as usize);
            unsafe {
                gl::GetShaderInfoLog(
                    id,
                    len,
                    std::ptr::null_mut(),
                    error.as_ptr() as *mut GLchar,
                );
            }
            Err(error.to_string_lossy().into_owned())
        } else {
            Ok(Shader { id })
        }
    }

    pub fn vertex_from_source(source: &CStr) -> Result<Shader, String> {
        Shader::from_source(source, gl::VERTEX_SHADER)
    }

    pub fn fragment_from_source(source: &CStr) -> Result<Shader, String> {
        Shader::from_source(source, gl::FRAGMENT_SHADER)
    }

    pub fn id(&self) -> GLuint {
        self.id
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe { gl::DeleteShader(self.id) };
    }
}

pub struct Program {
    id: GLuint,
}

impl Program {
    pub fn from_shaders(shaders: &[Shader]) -> Result<Program, String> {
        let id = unsafe { gl::CreateProgram() };

        for shader in shaders {
            unsafe { gl::AttachShader(id, shader.id()) };
        }

        unsafe { gl::LinkProgram(id) };

        let mut success: GLint = 1;
        unsafe {
            gl::GetProgramiv(id, gl::LINK_STATUS, &mut success);
        }

        if success == 0 {
            let mut len: GLint = 0;
            unsafe {
                gl::GetProgramiv(id, gl::INFO_LOG_LENGTH, &mut len);
            }
            let error = create_whitespace_cstring(len as usize);
            unsafe {
                gl::GetProgramInfoLog(
                    id,
                    len,
                    std::ptr::null_mut(),
                    error.as_ptr() as *mut GLchar,
                );
            }
            return Err(error.to_string_lossy().into_owned());
        }

        for shader in shaders {
            unsafe { gl::DetachShader(id, shader.id()) }
        }

        Ok(Program { id })
    }

    pub fn id(&self) -> GLuint {
        self.id
    }

    pub fn use_program(&self) {
        unsafe {
            gl::UseProgram(self.id);
        }
    }

    pub fn get_location(&self, name: &str) -> Result<GLint, String> {
        let cname = CString::new(name).unwrap();
        let loc = unsafe {
            gl::GetUniformLocation(self.id, cname.as_ptr() as *mut GLchar)
        };

        if loc != -1 {
            Ok(loc)
        } else {
            Err("Could not get location".to_owned())
        }
    }

    pub fn set_uniform(&self, loc: GLint, v: i32) {
        unsafe {
            gl::ProgramUniform1i(self.id, loc, v);
        }
    }
}

fn create_whitespace_cstring(len: usize) -> CString {
    let mut buffer: Vec<u8> = Vec::with_capacity(len as usize + 1);
    buffer.extend([b' '].iter().cycle().take(len as usize));
    unsafe { CString::from_vec_unchecked(buffer) }
}

#[repr(u32)]
#[derive(Copy, Clone)]
pub enum BufferType {
    ArrayBuffer = gl::ARRAY_BUFFER,
}

#[repr(u32)]
#[derive(Copy, Clone)]
pub enum BufferUsage {
    StaticDraw = gl::STATIC_DRAW,
    StreamDraw = gl::STREAM_DRAW,
}

pub struct Buffer<T> {
    id: GLuint,
    buffer_type: BufferType,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Buffer<T> {
    pub fn new(buffer_type: BufferType) -> Buffer<T> {
        let mut id: GLuint = 0;
        unsafe {
            gl::GenBuffers(1, &mut id);
        }
        Buffer {
            id,
            buffer_type,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn id(&self) -> GLuint {
        self.id
    }

    pub fn buffer_data(&self, data: &[T], usage: BufferUsage) {
        unsafe {
            gl::BindBuffer(self.buffer_type as GLuint, self.id);
            gl::BufferData(
                self.buffer_type as GLuint,
                (data.len() * std::mem::size_of::<T>()) as GLsizeiptr,
                data.as_ptr() as *const GLvoid,
                usage as GLenum,
            );
            gl::BindBuffer(self.buffer_type as GLuint, 0);
        }
    }

    pub fn bind(&self) {
        unsafe {
            gl::BindBuffer(self.buffer_type as GLuint, self.id);
        }
    }

    pub fn unbind(&self) {
        unsafe {
            gl::BindBuffer(self.buffer_type as GLuint, 0);
        }
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.id as *const GLuint);
        }
    }
}

pub struct VertexArray {
    id: GLuint,
}

impl VertexArray {
    pub fn new() -> VertexArray {
        let mut id: GLuint = 0;
        unsafe {
            gl::GenVertexArrays(1, &mut id);
        }

        VertexArray { id }
    }

    pub fn id(&self) -> GLuint {
        self.id
    }

    pub fn bind(&self) {
        unsafe {
            gl::BindVertexArray(self.id);
        }
    }

    pub fn unbind(&self) {
        unsafe {
            gl::BindVertexArray(0);
        }
    }
}

impl Drop for VertexArray {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteVertexArrays(1, &self.id as *const GLuint);
        }
    }
}

#[allow(non_camel_case_types)]
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct f32x2 {
    x: f32,
    y: f32,
}

impl f32x2 {
    pub fn new(x: f32, y: f32) -> f32x2 {
        f32x2 { x, y }
    }

    pub fn num_components() -> usize {
        2
    }
}

#[allow(non_camel_case_types)]
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct f32x3 {
    x: f32,
    y: f32,
    z: f32,
}

impl f32x3 {
    pub fn new(x: f32, y: f32, z: f32) -> f32x3 {
        f32x3 { x, y, z }
    }

    pub fn num_components() -> usize {
        3
    }
}

#[allow(non_camel_case_types)]
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct f32x4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl f32x4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> f32x4 {
        f32x4 { x, y, z, w }
    }

    pub fn zero() -> f32x4 {
        f32x4::new(0.0, 0.0, 0.0, 0.0)
    }

    pub fn set(&mut self, x: f32, y: f32, z: f32, w: f32) {
        self.x = x;
        self.y = y;
        self.z = z;
        self.w = w;
    }

    pub fn num_components() -> usize {
        4
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct Vertex {
    p: f32x3,
    st: f32x2,
}
impl Vertex {
    pub fn new(p: f32x3, st: f32x2) -> Vertex {
        Vertex { p, st }
    }

    unsafe fn vertex_attrib_pointer(
        num_components: usize,
        stride: usize,
        location: usize,
        offset: usize,
    ) {
        gl::EnableVertexAttribArray(location as gl::types::GLuint); // location(0)
        gl::VertexAttribPointer(
            location as gl::types::GLuint, // index of the vertex attribute
            num_components as gl::types::GLint, /* number of components per
                                            * vertex attrib */
            gl::FLOAT,
            gl::FALSE, // normalized (int-to-float conversion),
            stride as gl::types::GLint, /* byte stride between
                        * successive elements */
            offset as *const gl::types::GLvoid, /* offset of the first
                                                 * element */
        );
    }

    pub fn vertex_attrib_pointers() {
        let stride = std::mem::size_of::<Self>();

        let location = 0;
        let offset = 0;

        // and configure the vertex array
        unsafe {
            Vertex::vertex_attrib_pointer(
                f32x3::num_components(),
                stride,
                location,
                offset,
            );
        }

        let location = location + 1;
        let offset = offset + std::mem::size_of::<f32x3>();

        // and configure the st array
        unsafe {
            Vertex::vertex_attrib_pointer(
                f32x2::num_components(),
                stride,
                location,
                offset,
            );
        }
    }
}

pub struct FullscreenQuad {
    width: u32,
    height: u32,
    vertex_array: VertexArray,
    program: Program,
    texture_id: GLuint,
    loc_progression: GLint,
}

impl FullscreenQuad {
    pub fn new(width: u32, height: u32) -> Result<FullscreenQuad, String> {
        let vert_shader = Shader::vertex_from_source(
            CStr::from_bytes_with_nul(
                b"
            #version 330 core
            layout (location = 0) in vec3 _p;
            layout (location = 1) in vec2 _st;
            out vec2 st;
            void main() {
                gl_Position = vec4(_p, 1.0);
                st = _st;
            }
        \0",
            )
            .unwrap(),
        )?;

        let frag_shader = Shader::fragment_from_source(
            CStr::from_bytes_with_nul(
                b"
            #version 330 core
            in vec2 st;
            out vec4 Color;

            uniform sampler2D smp2d_0;
            uniform int progression;

            void main() {
                vec4 col = texture(smp2d_0, st); 
                col.r = pow(col.r / progression, 1/2.2);
                col.g = pow(col.g / progression, 1/2.2);
                col.b = pow(col.b / progression, 1/2.2);
                Color = col;
            }
        \0",
            )
            .unwrap(),
        )?;

        let program = Program::from_shaders(&[vert_shader, frag_shader])?;
        program.use_program();
        let loc_progression = program.get_location("progression")?;

        let vertices: Vec<Vertex> = vec![
            Vertex::new(f32x3::new(-1.0, -1.0, 0.0), f32x2::new(0.0, 0.0)),
            Vertex::new(f32x3::new(1.0, -1.0, 0.0), f32x2::new(1.0, 0.0)),
            Vertex::new(f32x3::new(1.0, 1.0, 0.0), f32x2::new(1.0, 1.0)),
            Vertex::new(f32x3::new(-1.0, -1.0, 0.0), f32x2::new(0.0, 0.0)),
            Vertex::new(f32x3::new(1.0, 1.0, 0.0), f32x2::new(1.0, 1.0)),
            Vertex::new(f32x3::new(-1.0, 1.0, 0.0), f32x2::new(0.0, 1.0)),
        ];
        let vertex_buffer = Buffer::<Vertex>::new(BufferType::ArrayBuffer);
        vertex_buffer.buffer_data(&vertices, BufferUsage::StaticDraw);

        // Generate and bind the VAO
        let vertex_array = VertexArray::new();

        vertex_array.bind();
        // Re-bind the VBO to associate the two. We could just have left it
        // bound earlier and let the association happen when we
        // configure the VAO but this way at least makes the connection
        // between the two seem more explicit, despite the magical
        // state machine hiding in OpenGL
        vertex_buffer.bind();

        // Set up the vertex attribute pointers for all locations
        Vertex::vertex_attrib_pointers();

        // now unbind both the vbo and vao to keep everything cleaner
        vertex_buffer.unbind();
        vertex_array.unbind();

        // generate test texture data using the image width rather than the
        // framebuffer width
        let mut tex_data = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                tex_data.push(f32x4::new(
                    (x as f32) / width as f32,
                    (y as f32) / height as f32,
                    1.0,
                    0.0,
                ));
            }
        }

        // generate the texture for the quad
        let mut texture_id: gl::types::GLuint = 0;
        unsafe {
            gl::GenTextures(1, &mut texture_id);
            gl::ActiveTexture(gl::TEXTURE0);
            gl::Enable(gl::TEXTURE_2D);
            gl::BindTexture(gl::TEXTURE_2D, texture_id);
            gl::TexParameteri(
                gl::TEXTURE_2D,
                gl::TEXTURE_WRAP_S,
                gl::CLAMP_TO_BORDER as gl::types::GLint,
            );
            gl::TexParameteri(
                gl::TEXTURE_2D,
                gl::TEXTURE_WRAP_T,
                gl::CLAMP_TO_BORDER as gl::types::GLint,
            );
            gl::TexParameteri(
                gl::TEXTURE_2D,
                gl::TEXTURE_MIN_FILTER,
                gl::NEAREST as gl::types::GLint,
            );
            gl::TexParameteri(
                gl::TEXTURE_2D,
                gl::TEXTURE_MAG_FILTER,
                gl::NEAREST as gl::types::GLint,
            );
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGBA32F as gl::types::GLint,
                width as gl::types::GLint,
                height as gl::types::GLint,
                0,
                gl::RGBA,
                gl::FLOAT,
                tex_data.as_ptr() as *const gl::types::GLvoid,
            );
        }

        Ok(FullscreenQuad {
            width,
            height,
            vertex_array,
            program,
            texture_id,
            loc_progression,
        })
    }

    pub fn draw(&self) {
        self.program.use_program();
        self.vertex_array.bind();
        unsafe {
            gl::DrawArrays(
                gl::TRIANGLES,
                0, // starting index in the enabled array
                6, // number of indices to draw
            )
        }
        self.vertex_array.unbind();
    }

    pub fn update_texture(&self, data: &[V4f32]) {
        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, self.texture_id);
            gl::TexSubImage2D(
                gl::TEXTURE_2D,
                0,
                0,
                0,
                self.width as GLint,
                self.height as GLint,
                gl::RGBA,
                gl::FLOAT,
                data.as_ptr() as *const GLvoid,
            );
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        unsafe {
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGBA32F as gl::types::GLint,
                width as gl::types::GLint,
                height as gl::types::GLint,
                0,
                gl::RGBA,
                gl::FLOAT,
                std::ptr::null(),
            );
        }
        self.width = width;
        self.height = height;
    }

    pub fn set_progression(&self, progression: i32) {
        self.program.set_uniform(self.loc_progression, progression);
    }
}
