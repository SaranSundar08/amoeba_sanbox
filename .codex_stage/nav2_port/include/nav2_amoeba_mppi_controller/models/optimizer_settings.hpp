// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NAV2_AMOEBA_MPPI_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_
#define NAV2_AMOEBA_MPPI_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_

#include <cstddef>
#include <string>
#include "nav2_amoeba_mppi_controller/models/constraints.hpp"

namespace amoeba::models
{

/**
 * @struct amoeba::models::OptimizerSettings
 * @brief Settings for the optimizer to use
 */
struct OptimizerSettings
{
  models::ControlConstraints base_constraints{0, 0, 0, 0};
  models::ControlConstraints constraints{0, 0, 0, 0};
  models::SamplingStd sampling_std{0, 0, 0};
  float model_dt{0};
  float temperature{0};
  float gamma{0};
  unsigned int batch_size{0};
  unsigned int time_steps{0};
  unsigned int iteration_count{0};
  bool shift_control_sequence{false};
  size_t retry_attempt_limit{0};

  // --- Amoeba shape-conditioned sampling (this controller's core behavior) ---
  float amoeba_bias_strength{0.6f};   // fraction of the batch seeded onto the wrap modes
  float amoeba_bias_gain{1.5f};       // heading P-gain: tangent bearing error -> yaw rate
  float amoeba_lookahead_dist{0.6f};  // path lookahead for the "toward-goal" bearing (m)
  float amoeba_scan_range{2.5f};      // forward ray-cast range into the local costmap (m)
  float amoeba_arc{1.57f};            // forward half-arc scanned each side (rad, ~90 deg)
  int amoeba_num_rays{37};            // angular resolution of the polar gap-scan
  float amoeba_goal_dist{1.0f};       // within this range of the goal, disable the amoeba
                                      // Let plain MPPI + GoalCritic dock near the goal.
  bool amoeba_debug{false};           // publish /amoeba_debug markers (scan rays + wrap arrows)
  bool amoeba_ancillary_debug{false};   // publish up to 3 lightweight Path candidates

  // Observe and visualize amoeba state without changing MPPI samples or costs.
  // This is the safe first Nav2 integration phase and defaults to enabled.
  bool amoeba_shadow_mode{true};
  // Independent Phase 3 authority gates. Shadow mode remains the master
  // override: when true, neither sampling nor critic scoring may affect MPPI.
  bool amoeba_bias_enabled{false};
  bool flow_critic_enabled{false};
  float amoeba_body_radius{2.5f};
  bool amoeba_debug_grid{false};
  float amoeba_debug_arrow_spacing{0.40f};
  float amoeba_debug_arrow_length{0.18f};
  float amoeba_debug_arrow_width{0.015f};

  // --- Amoeba flow mode: the water field over the local costmap ------------
  std::string amoeba_mode{"flow"};    // "flow" = water field bias (new core),
                                      // "ray"  = legacy polar gap-scan (kept for ablation)
  int flow_reflood_every{1};          // rebuild the field every N control cycles
  bool flow_path_seed{true};          // window-boundary promise from the plan
                                      // (dist-to-plan + remaining length) vs
                                      // straight-line-to-plan-end (planner-free)
  float flow_viscosity{1.5f};         // water physics: flood resistance near
                                      // walls (0 = pure geodesic); wide
                                      // channels flow freely, pinches drag
  float flow_promise_temperature{0.75f};  // softmax temperature (m) used to
                                          // allocate samples among pseudopods
  bool ancillary_collision_check{true};   // validate ancillary mean rollouts with the robot footprint
  int ancillary_collision_stride{1};      // footprint-check every Nth horizon pose (1 = every pose)
  bool flow_assist_only_when_path_blocked{true};  // preserve the global path in NORMAL mode
  float flow_path_check_distance{1.5f};           // local plan distance inspected by the gate (m)
  float flow_path_blocked_ratio{0.07f};           // enter ASSIST above this invalid-point ratio
  int flow_clear_confirm_cycles{3};               // clear cycles required before leaving ASSIST
  bool flow_wait_enabled{true};                   // offer an explicit zero-control sample population
  float flow_wait_fraction{0.20f};                // fraction of the biased budget reserved for waiting
  float flow_rejoin_lateral_weight{3.0f};         // penalize pseudopods far from the global path
  float flow_rejoin_remaining_weight{2.0f};       // penalize modes that lose path progress
};

}  // namespace amoeba::models

#endif  // NAV2_AMOEBA_MPPI_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_
