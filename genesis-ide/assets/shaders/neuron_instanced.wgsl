#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}

struct MaterialUniforms {
    base_color: vec4<f32>,
    clip_plane: vec4<f32>,
    view_mode: u32,
    _padding: vec3<f32>,
};

struct NeuronPalette {
    colors: array<vec4<f32>, 16>,
};

struct NeuronInstance {
    emissive: f32, // Deprecated, but kept for compatibility in instances buffer
    selected: u32,
}

@group(2) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(1) var<storage, read> selection: array<u32>;
@group(2) @binding(2) var<storage, read> geometry: array<u32>;
@group(2) @binding(3) var<storage, read> telemetry: array<f32>;
@group(2) @binding(4) var<uniform> palette: NeuronPalette;

struct Vertex {
    @location(0) position: vec3<f32>,
    @builtin(instance_index) instance_idx: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) glow: f32,
    @location(2) @interpolate(flat) selected: u32,
    @location(3) world_position: vec3<f32>,
}

@vertex
fn vertex(v: Vertex) -> VertexOutput {
    var out: VertexOutput;
    
    // O(1) аппаратная распаковка за 1 такт ALU
    let packed = geometry[v.instance_idx];
    let x = f32(packed & 0x7FFu);
    let y = f32((packed >> 11u) & 0x7FFu);
    let z = f32((packed >> 22u) & 0x3Fu);
    let type_id = (packed >> 28u) & 0xFu; 
    
    // Получаем цвет из палитры за O(1)
    let type_color = palette.colors[type_id];
    
    // Распаковка флага выделения (Broadcast Read + Shift)
    let word_idx = v.instance_idx / 32u;
    let bit_idx = v.instance_idx % 32u;
    out.selected = (selection[word_idx] >> bit_idx) & 1u;
    
    out.glow = telemetry[v.instance_idx];
    
    let voxel_size = 25.0; // Voxel: 25um
    let local_neuron_pos = vec3<f32>(x, y, z) * voxel_size;
    
    let final_pos = v.position + local_neuron_pos;
    let world_from_local = get_world_from_local(v.instance_idx);
    
    out.world_position = (world_from_local * vec4<f32>(final_pos, 1.0)).xyz;
    out.clip_position = mesh_position_local_to_clip(world_from_local, vec4<f32>(final_pos, 1.0));
    out.color = type_color;
    
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Clipping
    if dot(in.world_position, material.clip_plane.xyz) > material.clip_plane.w {
        discard;
    }

    var final_color: vec3<f32> = in.color.rgb;

    // Режим Activity
    if material.view_mode == 1u {
        let cold = vec3<f32>(0.02, 0.05, 0.15);
        let warm = vec3<f32>(0.8, 0.1, 0.0);
        let hot = vec3<f32>(1.0, 0.9, 0.2);

        // Используем glow (телеметрию) для термокарты
        var heat = mix(cold, warm, smoothstep(0.0, 0.4, in.glow));
        heat = mix(heat, hot, smoothstep(0.4, 1.0, in.glow));
        final_color = heat;
    }

    // HDR свечение спайка
    let emissive_color = vec3<f32>(1.0, 0.3, 0.1); 
    let spike_glow = emissive_color * in.glow * 25.0;
    final_color += spike_glow;

    // Подсветка выделения
    if in.selected > 0u {
        final_color = mix(final_color, vec3<f32>(0.2, 0.8, 1.0), 0.6);
    }

    return vec4<f32>(final_color, 1.0);
}
