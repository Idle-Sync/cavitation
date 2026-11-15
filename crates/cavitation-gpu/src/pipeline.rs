//! Compute pipelines built once per shader, entry point and set of `override` values, and
//! reused from then on; the cache counts what it builds, so a test can check that warm
//! steps build nothing.

use std::collections::HashMap;

use crate::{Gpu, GpuError};

/// A WGSL compute shader: its source and a name unique among the shaders given to one
/// [`PipelineCache`], used as the cache key and in error messages.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::Shader;
///
/// const DOUBLE: Shader = Shader {
///     name: "double",
///     source: "@group(0) @binding(0) var<storage, read_write> x: array<f32>;
///              @compute @workgroup_size(64)
///              fn main(@builtin(global_invocation_id) id: vec3<u32>) {
///                  if (id.x < arrayLength(&x)) { x[id.x] = 2.0 * x[id.x]; }
///              }",
/// };
/// assert_eq!(DOUBLE.name, "double");
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Shader {
    /// The shader's name, unique among the shaders given to one cache.
    pub name: &'static str,
    /// The WGSL source.
    pub source: &'static str,
}

/// What identifies one pipeline in a [`PipelineCache`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PipelineKey {
    /// The shader's name.
    shader: &'static str,
    /// The entry point.
    entry: String,
    /// The `override` values by name, sorted by name, each as the bits of its `f64`.
    constants: Vec<(String, u64)>,
    /// The explicit pipeline layout, or `None` for the one wgpu derives from the shader.
    layout: Option<wgpu::PipelineLayout>,
}

/// Compute pipelines by shader, entry point and `override` values, each built once.
///
/// [`get`](PipelineCache::get) builds a pipeline the first time a key is asked for (and the
/// shader module the first time a shader is) and returns the same pipeline afterwards. The
/// `override` values are part of the key, in any order: the same shader specialised two
/// ways is two pipelines.
///
/// With [`get`](PipelineCache::get), wgpu derives each pipeline's layout from the shader;
/// its bind groups are made with `pipeline.get_bind_group_layout(group)` and serve that
/// pipeline only. [`get_with_layout`](PipelineCache::get_with_layout) takes an explicit
/// layout instead, so one bind group serves every pipeline built with that layout, such as
/// one shader specialised several ways.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::{Gpu, PipelineCache, Shader};
///
/// const FILL: Shader = Shader {
///     name: "fill",
///     source: "override VALUE: f32 = 1.0;
///              @group(0) @binding(0) var<storage, read_write> x: array<f32>;
///              @compute @workgroup_size(64)
///              fn main(@builtin(global_invocation_id) id: vec3<u32>) {
///                  if (id.x < arrayLength(&x)) { x[id.x] = VALUE; }
///              }",
/// };
/// if let Some(gpu) = Gpu::new() {
///     let mut cache = PipelineCache::new(&gpu);
///     let ones = cache.get(FILL, "main", &[]).unwrap();
///     let twos = cache.get(FILL, "main", &[("VALUE", 2.0)]).unwrap();
///     let again = cache.get(FILL, "main", &[]).unwrap();
///     assert_eq!(cache.pipelines_built(), 2);
///     assert_eq!(cache.modules_built(), 1);
///     # drop((ones, twos, again));
/// }
/// ```
#[derive(Debug)]
pub struct PipelineCache {
    /// The GPU the pipelines are built on.
    gpu: Gpu,
    /// The source each shader name has been given with, recorded at its first use, built
    /// or not.
    sources: HashMap<&'static str, &'static str>,
    /// The shader modules by shader name, for the shaders that have built a pipeline.
    modules: HashMap<&'static str, wgpu::ShaderModule>,
    /// The pipelines built so far.
    pipelines: HashMap<PipelineKey, wgpu::ComputePipeline>,
    /// The number of shader modules built so far.
    modules_built: usize,
    /// The number of pipelines built so far.
    pipelines_built: usize,
}

impl PipelineCache {
    /// An empty cache on `gpu`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     assert_eq!(PipelineCache::new(&gpu).pipelines_built(), 0);
    /// }
    /// ```
    pub fn new(gpu: &Gpu) -> PipelineCache {
        PipelineCache {
            gpu: gpu.clone(),
            sources: HashMap::new(),
            modules: HashMap::new(),
            pipelines: HashMap::new(),
            modules_built: 0,
            pipelines_built: 0,
        }
    }

    /// The pipeline for `entry` in `shader` with the `override` values `constants` (in any
    /// order), built the first time it is asked for.
    ///
    /// A shader that does not compile (in naga or in the backend's own compiler), a
    /// missing entry point and an unknown, ill-typed or NaN `override` are
    /// [`GpuError::Shader`], with wgpu's message; nothing is cached for them, so the next
    /// call tries again.
    ///
    /// # Panics
    ///
    /// If another shader with the same name but a different source was given to this cache
    /// before.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, GpuError, PipelineCache, Shader};
    ///
    /// const BROKEN: Shader = Shader { name: "broken", source: "fn main( {" };
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut cache = PipelineCache::new(&gpu);
    ///     assert!(matches!(cache.get(BROKEN, "main", &[]), Err(GpuError::Shader { .. })));
    /// }
    /// ```
    pub fn get(
        &mut self,
        shader: Shader,
        entry: &str,
        constants: &[(&str, f64)],
    ) -> Result<wgpu::ComputePipeline, GpuError> {
        self.build(shader, entry, constants, None)
    }

    /// As [`get`](PipelineCache::get), with an explicit pipeline layout, which is part of
    /// the key. Bind groups made from the layout's bind group layouts serve every pipeline
    /// built with it.
    ///
    /// A layout that does not match the shader's bindings is [`GpuError::Shader`].
    ///
    /// # Panics
    ///
    /// As [`get`](PipelineCache::get).
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, PipelineCache, Shader, wgpu};
    ///
    /// const FILL: Shader = Shader {
    ///     name: "fill",
    ///     source: "override VALUE: f32 = 1.0;
    ///              @group(0) @binding(0) var<storage, read_write> x: array<f32>;
    ///              @compute @workgroup_size(64)
    ///              fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    ///                  if (id.x < arrayLength(&x)) { x[id.x] = VALUE; }
    ///              }",
    /// };
    /// if let Some(gpu) = Gpu::new() {
    ///     let device = gpu.device();
    ///     let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    ///         label: None,
    ///         entries: &[wgpu::BindGroupLayoutEntry {
    ///             binding: 0,
    ///             visibility: wgpu::ShaderStages::COMPUTE,
    ///             ty: wgpu::BindingType::Buffer {
    ///                 ty: wgpu::BufferBindingType::Storage { read_only: false },
    ///                 has_dynamic_offset: false,
    ///                 min_binding_size: None,
    ///             },
    ///             count: None,
    ///         }],
    ///     });
    ///     let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    ///         label: None,
    ///         bind_group_layouts: &[Some(&group)],
    ///         immediate_size: 0,
    ///     });
    ///     let mut cache = PipelineCache::new(&gpu);
    ///     // Both pipelines take bind groups made from `group`.
    ///     let ones = cache.get_with_layout(FILL, "main", &[], &layout).unwrap();
    ///     let twos = cache.get_with_layout(FILL, "main", &[("VALUE", 2.0)], &layout).unwrap();
    ///     # drop((ones, twos));
    /// }
    /// ```
    pub fn get_with_layout(
        &mut self,
        shader: Shader,
        entry: &str,
        constants: &[(&str, f64)],
        layout: &wgpu::PipelineLayout,
    ) -> Result<wgpu::ComputePipeline, GpuError> {
        self.build(shader, entry, constants, Some(layout))
    }

    /// The cached pipeline for the key, or a newly built one: [`get`](PipelineCache::get)
    /// and [`get_with_layout`](PipelineCache::get_with_layout) both come here.
    fn build(
        &mut self,
        shader: Shader,
        entry: &str,
        constants: &[(&str, f64)],
        layout: Option<&wgpu::PipelineLayout>,
    ) -> Result<wgpu::ComputePipeline, GpuError> {
        let source = *self.sources.entry(shader.name).or_insert(shader.source);
        assert!(
            source == shader.source,
            "two different shaders are named {:?}",
            shader.name
        );
        let key = key(shader, entry, constants, layout);
        if let Some(pipeline) = self.pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let device = self.gpu.device();
        // Naga's and the pipeline's checks are validation errors; a backend compiler's
        // failure (FXC, DXC, the Metal or Vulkan driver) is an internal one.
        let scopes = [
            wgpu::ErrorFilter::OutOfMemory,
            wgpu::ErrorFilter::Internal,
            wgpu::ErrorFilter::Validation,
        ]
        .map(|filter| device.push_error_scope(filter));
        let module = match self.modules.get(shader.name) {
            Some(module) => module.clone(),
            None => device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(shader.name),
                source: wgpu::ShaderSource::Wgsl(shader.source.into()),
            }),
        };
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(shader.name),
            layout,
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            cache: None,
        });
        // Scopes pop innermost first: validation, internal, out of memory.
        let mut errors = Vec::new();
        for scope in scopes.into_iter().rev() {
            if let Some(error) = pollster::block_on(scope.pop()) {
                errors.push(error.to_string());
            }
        }
        if !errors.is_empty() {
            return Err(GpuError::Shader {
                shader: format!("{} ({entry})", shader.name),
                message: errors.join("; "),
            });
        }
        if !self.modules.contains_key(shader.name) {
            self.modules.insert(shader.name, module);
            self.modules_built += 1;
        }
        self.pipelines_built += 1;
        self.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// The number of pipelines this cache has built.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     assert_eq!(PipelineCache::new(&gpu).pipelines_built(), 0);
    /// }
    /// ```
    pub fn pipelines_built(&self) -> usize {
        self.pipelines_built
    }

    /// The number of shader modules this cache has built: one per shader that has built a
    /// pipeline.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     assert_eq!(PipelineCache::new(&gpu).modules_built(), 0);
    /// }
    /// ```
    pub fn modules_built(&self) -> usize {
        self.modules_built
    }
}

/// The cache key for `entry` in `shader` with `constants` and `layout`, the constants sorted
/// by name.
fn key(
    shader: Shader,
    entry: &str,
    constants: &[(&str, f64)],
    layout: Option<&wgpu::PipelineLayout>,
) -> PipelineKey {
    let mut constants: Vec<(String, u64)> = constants
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_bits()))
        .collect();
    constants.sort();
    PipelineKey {
        shader: shader.name,
        entry: entry.to_string(),
        constants,
        layout: layout.cloned(),
    }
}

#[cfg(test)]
mod tests {
    //! The cache key; the cache itself is tested on a GPU in `tests/pool_and_cache.rs`.

    use super::*;

    /// A shader for the keys; it is never compiled here.
    const SHADER: Shader = Shader {
        name: "s",
        source: "",
    };

    /// The order of the `override` values does not change the key; their values, the entry
    /// point and the shader do.
    #[test]
    fn keys() {
        let a = key(SHADER, "main", &[("A", 1.0), ("B", 2.0)], None);
        assert_eq!(a, key(SHADER, "main", &[("B", 2.0), ("A", 1.0)], None));
        assert_ne!(a, key(SHADER, "main", &[("A", 1.0), ("B", 3.0)], None));
        assert_ne!(a, key(SHADER, "other", &[("A", 1.0), ("B", 2.0)], None));
        let other = Shader {
            name: "t",
            source: "",
        };
        assert_ne!(a, key(other, "main", &[("A", 1.0), ("B", 2.0)], None));
        assert_ne!(
            key(SHADER, "main", &[("A", 0.0)], None),
            key(SHADER, "main", &[("A", -0.0)], None)
        );
    }
}
