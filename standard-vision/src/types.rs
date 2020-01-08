use crate::traits::ImageData;
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    time::SystemTime,
};

/// General struct for storing positional data
pub struct Pose {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub angle: f64,
}

/// Description of a camera's properties
pub struct CameraConfig {
    pub id: u8,
    pub resolution: (u32, u32),
    pub pose: Pose,
    pub fov: f64,
    pub focal_length: f64,
    pub sensor_height: f64,
}


/// `Image` stores an images data as well as its associated properties and derefs into `pixels`.
/// `T` is the raw data type of the image in memory.
pub struct Image<'a, T, I: ImageData<T>> {
    pub timestamp: SystemTime,
    pub camera: &'a CameraConfig,
    pub pixels: I,
    raw: PhantomData<T>,
}

impl<'a, T, I: ImageData<T>> Image<'a, T, I> {
    pub fn new(timestamp: SystemTime, camera: &'a CameraConfig, pixels: I) -> Self {
        Self {
            timestamp,
            camera,
            pixels,
            raw: PhantomData,
        }
    }
}

impl<'a, T, I: ImageData<T>> Deref for Image<'a, T, I> {
    type Target = I;

    fn deref(&self) -> &Self::Target {
        &self.pixels
    }
}

impl<'a, T, I: ImageData<T>> DerefMut for Image<'a, T, I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.pixels
    }
}

/// Stores contours of an images.
pub struct Contour {
    pub points: Vec<(u32, u32)>,
}
