use crate::traits::ImageData;
use std::{
    ops::{Deref, DerefMut},
    rc::Rc,
    time::SystemTime,
};

/// A representation of a relative position and rotation.
#[derive(Default)]
pub struct Pose {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub angle: f64,
}

/// A collection of camera properties.
#[derive(Default)]
pub struct CameraConfig {
    pub id: u8,
    pub resolution: (u32, u32),
    pub pose: Pose,
    pub fov: (f64, f64),
    pub focal_length: f64,
    pub sensor_height: f64,
}


/// An image, backed by a generic image data type `I`.
pub struct Image<I: ImageData> {
    pub timestamp: SystemTime,
    pub camera: &'a CameraConfig,
    pub pixels: I,
}

impl<I: ImageData> Image<I> {
    pub fn new(timestamp: SystemTime, camera: Rc<CameraConfig>, pixels: I) -> Self {
        Self {
            timestamp,
            camera,
            pixels,
        }
    }
}

impl<I: ImageData> Deref for Image<I> {
    type Target = I;

    fn deref(&self) -> &Self::Target {
        &self.pixels
    }
}

impl<I: ImageData> DerefMut for Image<I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.pixels
    }
}

/// A collection of points that form a contour.
pub struct Contour {
    pub points: Vec<(u32, u32)>,
}

pub struct Target {
    pub camera: Rc<CameraConfig>,
    pub contours: Vec<Contour>,
    pub theta: f64,
    pub beta: f64,
    pub dist: f64,
    pub confidence: f32,
}
