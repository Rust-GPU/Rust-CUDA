use crate::*;

const BACKGROUND_BLUE_MULTIPLIER: f32 = 0.7;

pub fn color(ray: Ray) -> Vec3 {
    let unit = ray.dir.normalized();
    let t = BACKGROUND_BLUE_MULTIPLIER * (unit.y + 1.0);
    (1.0 - t) * Vec3::one() + t * Vec3::new(0.5, 0.7, 1.0)
}

pub fn generate_ray(idx: vek::Vec2<u32>, view: &Viewport, offset: Vec2) -> Ray {
    let uv = (idx.numcast::<f32>().unwrap() + offset) / view.bounds.numcast().unwrap();
    Ray {
        origin: view.origin,
        dir: view.lower_left + uv.x * view.horizontal + uv.y * view.vertical - view.origin,
    }
}
