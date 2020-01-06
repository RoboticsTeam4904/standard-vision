use std::ops::DerefMut;

type Pixels<'a> = ndarray::ArrayView3<'a, u8>;
type ImageData<'a> = Box<dyn DerefMut<Target = Pixels<'a>>>;

pub struct Pose {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub angle: f64,
}

pub struct CameraConfig {
    pub id: u8,
    pub resolution: (u32, u32),
    pub pose: Pose,
    pub fov: f64,
}


pub struct Image<'a> {
    pub timestamp: std::time::SystemTime,
    pub camera: &'a CameraConfig,
    pub pixels: ImageData<'a>,
}

pub struct Contour {
    pub points: Vec<(u32, u32)>,
}

pub struct Target<'a> {
    pub camera: &'a CameraConfig,
    pub contours: Vec<Contour>,
    pub theta: f64,
    pub beta: f64,
    pub dist: f64,
    pub confidence: f32,
}
