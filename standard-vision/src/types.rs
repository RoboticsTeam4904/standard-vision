use crate::traits::ImageData;
use std::{
    ops::{Deref, DerefMut},
    time::SystemTime,
};

/// A representation of a relative position and rotation.
pub struct Pose {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub angle: f64,
}

/// A collection of camera properties.
pub struct CameraConfig {
    pub id: u8,
    pub resolution: (u32, u32),
    pub pose: Pose,
    pub fov: (f64, f64),
    pub focal_length: f64,
    pub sensor_height: f64,
}


/// An image, backed by a generic image data type `I`.
pub struct Image<'a, I: ImageData> {
    pub timestamp: SystemTime,
    pub camera: &'a CameraConfig,
    pub pixels: I,
}

impl<'a, I: ImageData> Image<'a, I> {
    pub fn new(timestamp: SystemTime, camera: &'a CameraConfig, pixels: I) -> Self {
        Self {
            timestamp,
            camera,
            pixels,
        }
    }
}

impl<'a, I: ImageData> Deref for Image<'a, I> {
    type Target = I;

    fn deref(&self) -> &Self::Target {
        &self.pixels
    }
}

impl<'a, I: ImageData> DerefMut for Image<'a, I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.pixels
    }
}

/// A collection of points that form a contour.
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
