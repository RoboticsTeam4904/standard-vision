use crate::traits::ImageData;
use std::{ops::{Deref, DerefMut}, marker::PhantomData, time::SystemTime};

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
    pub focal_length: f64,
}

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
            raw: PhantomData::default(),
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
