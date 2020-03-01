#[cfg(test)]
use png;

use std::io;

use ndarray::{ArrayViewD, ArrayViewMutD};
use opencv::{prelude::*, videoio::*};
use stdvis_core::{
    traits::{Camera, ImageData},
    types::{CameraConfig, Image},
};

use crate::convert::AsArrayView;

pub struct MatImageData {
    mat: Mat,
}

impl MatImageData {
    pub fn new(mat: Mat) -> Self {
        Self { mat }
    }
}

impl ImageData for MatImageData {
    type Inner = Mat;

    fn as_pixels(&self) -> ArrayViewD<u8> {
        self.mat.as_array_view()
    }

    fn as_pixels_mut(&mut self) -> ArrayViewMutD<u8> {
        self.mat.as_array_view_mut()
    }

    fn as_raw(&self) -> &Self::Inner {
        &self.mat
    }

    fn as_raw_mut(&mut self) -> &mut Self::Inner {
        &mut self.mat
    }
}

pub struct OpenCVCamera {
    config: CameraConfig,
    video_capture: VideoCapture,
}

impl OpenCVCamera {
    pub fn new(config: CameraConfig) -> opencv::Result<Self> {
        let id = config.id;
        let mut video_capture = VideoCapture::default()?;

        video_capture.open_with_backend(id as i32, {
            if cfg!(target_os = "linux") {
                CAP_V4L
            } else {
                CAP_ANY
            }
        })?;

        video_capture.set(CAP_PROP_FRAME_WIDTH, config.resolution.0 as f64)?;
        video_capture.set(CAP_PROP_FRAME_HEIGHT, config.resolution.1 as f64)?;

        Ok(Self {
            config,
            video_capture,
        })
    }

    pub fn exposure(&self) -> opencv::Result<f64> {
        self.video_capture.get(CAP_PROP_EXPOSURE)
    }

    pub fn set_exposure(&mut self, exposure: f64) -> opencv::Result<bool> {
        self.video_capture.set(CAP_PROP_AUTO_EXPOSURE, 1.)?;
        self.video_capture.set(CAP_PROP_EXPOSURE, exposure)
    }
}

impl Camera<MatImageData> for OpenCVCamera {
    fn config(&self) -> &CameraConfig {
        &self.config
    }

    fn grab_frame(&mut self) -> io::Result<Image<MatImageData>> {
        let mut mat = Mat::default().unwrap();
        if !self.video_capture.read(&mut mat).unwrap() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Failed to read frame from camera at port {}",
                    self.config.id
                ),
            ));
        }

        Ok(Image::new(
            std::time::Instant::now(),
            self.config(),
            MatImageData::new(mat),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_pixel_extraction() {
        use ndarray::s;
        use opencv::{core::Vec3, imgcodecs};
        use std::fs::File;

        use crate::convert::AsMatView;

        const PATH: &str = "tests/images/rand.png";

        let decoder = png::Decoder::new(File::open(PATH).unwrap());
        let (info, mut reader) = decoder.read_info().unwrap();
        let mut img_buf = vec![0; info.buffer_size()];
        reader.next_frame(&mut img_buf).unwrap();

        let expected_pixels_buf = img_buf
            .chunks_exact(3)
            .map(|items| [items[2], items[1], items[0]])
            .collect::<Vec<_>>();

        let cv_image = MatImageData::new(imgcodecs::imread(PATH, imgcodecs::IMREAD_COLOR).unwrap());

        let config = CameraConfig::default();
        let image = Image::new(std::time::Instant::now(), &config, cv_image);

        let cv_pixels = image.as_pixels();
        let cv_raw = &image.as_mat_view();

        assert_eq!(
            cv_pixels.shape(),
            [
                cv_raw.rows().unwrap() as usize,
                cv_raw.cols().unwrap() as usize,
                cv_raw.channels().unwrap() as usize,
            ]
        );

        let img_dims = cv_pixels.shape();
        for row in 0..img_dims[0] {
            for col in 0..img_dims[1] {
                let expected_pixel = expected_pixels_buf[row * img_dims[1] + col];

                assert_eq!(
                    cv_pixels.slice(s![row, col, ..]).as_slice().unwrap(),
                    expected_pixel
                );

                assert_eq!(
                    **cv_raw.at_2d::<Vec3<u8>>(row as i32, col as i32).unwrap(),
                    expected_pixel
                );
            }
        }
    }
}
