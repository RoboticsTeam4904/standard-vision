use std::{
    ops::{Deref, DerefMut},
    time::Instant,
};

use mincodec::MinCodec;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::traits::ImageData;

/// A representation of a relative position and rotation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Pose {
    pub angle: f64,
    pub dist: f64,
    pub height: f64,
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
}

/// A collection of camera properties.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CameraConfig {
    pub id: u8,
    pub resolution: (u32, u32),
    pub pose: Pose,
    pub fov: (f64, f64),
    pub intrinsic_matrix: Array2<f64>,
    pub distortion_coeffs: Array1<f64>,
}

/// An image, backed by a generic image data type `I`.
pub struct Image<'src, Storage: ImageData> {
    pub timestamp: Instant,
    pub camera: &'src CameraConfig,
    pub pixels: Storage,
}

impl<'src, I: ImageData> Image<'src, I> {
    pub fn new(timestamp: Instant, camera: &'src CameraConfig, pixels: I) -> Self {
        Self {
            timestamp,
            camera,
            pixels,
        }
    }
}

impl<'src, I: ImageData> Deref for Image<'src, I> {
    type Target = I;

    fn deref(&self) -> &Self::Target {
        &self.pixels
    }
}

impl<'src, I: ImageData> DerefMut for Image<'src, I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.pixels
    }
}

/// A collection of points that form a contour.
#[derive(Debug)]
pub struct Contour {
    pub points: Vec<(f32, f32)>,
}

/// A collection of contours that form a logical group.
#[derive(Debug)]
pub struct ContourGroup<'src> {
    pub id: u8,
    pub camera: &'src CameraConfig,
    pub contours: Vec<Contour>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, MinCodec, PartialEq)]
pub struct VisionTarget {
    pub id: u8,
    pub beta: f64,
    pub theta: f64,
    pub dist: f64,
    pub height: f64,
    pub confidence: f32,
}
