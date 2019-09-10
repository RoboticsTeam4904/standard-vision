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
    pub camera: &'a Camera,
    pub pixels: Vec<(u8, u8, u8)>,
}

pub struct Contour<'a> {
    pub image: &'a Image<'a>,
    pub points: Vec<u32>,
}

pub struct Target<'a> {
    pub camera: &'a Camera,
    pub contours: Vec<Contour<'a>>,
    pub theta: f64,
    pub beta: f64,
    pub dist: f64,
    pub confidence: f32,
}
