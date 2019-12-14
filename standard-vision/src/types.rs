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
    pub pixels: Vec<[u8; 3]>,
}

impl<'a> Image<'a> {
    pub fn get(&self, y: usize, x: usize) -> [u8; 3] {
        self.pixels[y * self.camera.resolution.0 as usize + x]
    }
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
