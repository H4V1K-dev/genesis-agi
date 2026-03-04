use bevy::prelude::*;
use crate::{
    hud::SelectionState,
    layout::{AreaBody, EditorType},
    config_editor::LoadedConfig,
};

#[derive(Component)]
pub struct InspectorDataBinding;

pub struct NeuronInspectorPlugin;

impl Plugin for NeuronInspectorPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (
            build_inspector_ui,
            update_inspector_ui,
        ).chain());
    }
}

/// Строит UI один раз при открытии панели
fn build_inspector_ui(
    mut commands: Commands,
    q_bodies: Query<(Entity, &AreaBody), Added<AreaBody>>,
) {
    for (entity, body) in q_bodies.iter() {
        if body.0 != EditorType::NeuronInspector { continue; }

        commands.entity(entity).with_children(|parent| {
            parent.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    padding: UiRect::all(Val::Px(15.0)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.08, 0.08, 0.08)),
            )).with_children(|panel| {
                panel.spawn((
                    Text::new("No neuron selected.\nClick on a sphere in 3D View."),
                    TextFont { font_size: 16.0, ..default() },
                    TextColor(Color::srgb(0.8, 0.8, 0.8)),
                    InspectorDataBinding, // Маркер для Hot Loop обновления
                ));
            });
        });
    }
}

use crate::loader::LoadedGeometry;

/// Zero-Cost обновление. Выполняется только если изменился SelectionState.
fn update_inspector_ui(
    selection: Res<SelectionState>,
    config: Option<Res<LoadedConfig>>,
    geometry: Option<Res<LoadedGeometry>>,
    mut q_text: Query<&mut Text, With<InspectorDataBinding>>,
) {
    // Встроенный Change Detection Bevy. Если пользователь не кликал, выходим.
    if !selection.is_changed() { return; }

    let Some(config) = config else { return };
    let Some(geometry) = geometry else { return };
    let Some(blueprints) = &config.blueprints else { return };
    let Ok(mut text) = q_text.get_single_mut() else { return };

    // Берём первый нейрон из выделения
    if let Some(&(t_id, l_idx)) = selection.selected_neurons.first() {
        // В новой унифицированной архитектуре l_idx и есть global_idx
        let global_idx = l_idx;

        let packed = geometry.0[global_idx as usize];
        
        // Unpack according to Spec 03 §1.3
        let x = (packed & 0x7FFu32) as f32;
        let y = ((packed >> 11) & 0x7FFu32) as f32;
        let z = ((packed >> 22) & 0x3Fu32) as f32;
        let _p_type_id = ((packed >> 28) & 0xFu32) as u8;

        // Берем физические параметры из загруженного Blueprint
        let profile = blueprints.neuron_types.get(t_id as usize);
        let thresh = profile.map_or(0, |p| p.threshold);
        let rest = profile.map_or(0, |p| p.rest_potential);

        let new_str = format!(
            "=== NEURON INSPECTOR ===\n\n\
            Global Index: {}\n\
            Local Index:  {}\n\
            Type Index:   {} (4-bit mask)\n\n\
            [ Spatial Data ]\n\
            Position:     X:{:.1}  Y:{:.1}  Z:{:.1} um\n\n\
            [ Membrane Physics (Live Config) ]\n\
            Threshold:    {} mV\n\
            Rest Pot.:    {} mV\n",
            global_idx, l_idx, t_id, x, y, z, thresh, rest
        );

        text.0 = new_str;
    } else {
        text.0 = "No neuron selected.\nClick on a sphere in 3D View.".to_string();
    }
}

