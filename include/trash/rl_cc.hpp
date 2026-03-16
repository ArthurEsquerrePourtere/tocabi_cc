#include "p73_lib/robot_data.h"
#include "p73_lib/p73.h"
#include "rclcpp/rclcpp.hpp"
#include "onnxruntime_cxx_api.h"
#include <unordered_map>
#include <fstream>
#include <sensor_msgs/msg/joy.hpp>

class RlController
{
public:
    RlController(DataContainer &dc);
    ~RlController();

    double hz_ = 125.0;  // Default inference frequency, can be overridden by config

    void computeFast(); // for general case, use this 
    void computeSlow(); 
    void computeMpc();
    void copyRobotData(DataContainer &dc_global_);

    //NOTE - in order of computation
    DataContainer &dc_;
    DataContainer dc_cc_; // copy rd to rd_cc_ and use it locally

    void initVariablesAndLoadConfig(const std::string& config_file_path);  // Single cohesive initialization function
    void initBias();
    void loadOnnxModel(const std::map<std::string, std::string>& model_paths);
    
    Ort::Session* getSession(const std::string& name);
    size_t computeElementCount(const std::vector<int64_t>& shape);

    // Validation helper functions - marked inline for better optimization
    /**
     * @brief Validate and fix NaN/Inf values in a raw float array
     * @param data Pointer to float array
     * @param size Number of elements to check
     * @param default_val Value to replace NaN/Inf with
     * @return Index of first invalid value, or -1 if all valid
     * @note Marked inline and force-inlined for performance-critical path
     */
    __attribute__((always_inline)) inline int validateAndFixNaN(float* data, size_t size, float default_val = 0.0f) {
        int first_invalid_idx = -1;
        int nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < size; ++i) {
            if (!std::isfinite(data[i])) {
                if (std::isnan(data[i])) {
                    ++nan_count;
                    std::cerr << "[validateAndFixNaN] NaN detected at index " << i << ", replacing with " << default_val << std::endl;
                } else {
                    ++inf_count;
                    std::cerr << "[validateAndFixNaN] Inf detected at index " << i << " (value: " << data[i] << "), replacing with " << default_val << std::endl;
                }
                if (first_invalid_idx == -1) first_invalid_idx = static_cast<int>(i);
                data[i] = default_val;
            }
        }
        if (nan_count > 0 || inf_count > 0) {
            std::cerr << "[validateAndFixNaN] float* array: " << nan_count << " NaN, " << inf_count << " Inf detected (total size: " << size << ")" << std::endl;
        }
        return first_invalid_idx;
    }
    
    // Overload for std::vector - delegates to raw pointer version
    inline int validateAndFixNaN(std::vector<float>& data, size_t size, float default_val = 0.0f) {
        return validateAndFixNaN(data.data(), std::min(size, data.size()), default_val);
    }
    
    /**
     * @brief Validate and reset Eigen vector if NaN/Inf detected
     * @param data Eigen vector to validate
     * @param fallback Fallback vector to use if data contains NaN/Inf
     * @return true if data was reset, false otherwise
     */
    template<typename Derived1, typename Derived2>
    bool validateAndFixNaN(Eigen::MatrixBase<Derived1>& data, const Eigen::MatrixBase<Derived2>& fallback) {
        if (!data.allFinite()) {
            int nan_count = 0, inf_count = 0;
            for (int i = 0; i < data.size(); ++i) {
                if (std::isnan(data(i))) {
                    ++nan_count;
                    std::cerr << "[validateAndFixNaN] Eigen NaN at index " << i << std::endl;
                } else if (std::isinf(data(i))) {
                    ++inf_count;
                    std::cerr << "[validateAndFixNaN] Eigen Inf at index " << i << " (value: " << data(i) << ")" << std::endl;
                }
            }
            std::cerr << "[validateAndFixNaN] Eigen vector (with fallback): " << nan_count << " NaN, " << inf_count << " Inf detected, resetting to fallback" << std::endl;
            data = fallback;
            return true;
        }
        return false;
    }
    
    /**
     * @brief Validate and zero Eigen vector if NaN/Inf detected
     * @param data Eigen vector to validate  
     * @return true if data was zeroed, false otherwise
     */
    template<typename Derived>
    bool validateAndFixNaN(Eigen::MatrixBase<Derived>& data) {
        if (!data.allFinite()) {
            int nan_count = 0, inf_count = 0;
            for (int i = 0; i < data.size(); ++i) {
                if (std::isnan(data(i))) {
                    ++nan_count;
                    std::cerr << "[validateAndFixNaN] Eigen NaN at index " << i << std::endl;
                } else if (std::isinf(data(i))) {
                    ++inf_count;
                    std::cerr << "[validateAndFixNaN] Eigen Inf at index " << i << " (value: " << data(i) << ")" << std::endl;
                }
            }
            std::cerr << "[validateAndFixNaN] Eigen vector (zero fallback): " << nan_count << " NaN, " << inf_count << " Inf detected, zeroing" << std::endl;
            data.setZero();
            return true;
        }
        return false;
    }

    // Performance-critical functions - inline candidates
    void processNoise();
    void processBias();
    void processObservation();

    //SECTION - per ctrl frequency (hot path - consider inlining)
    void feedforwardPolicy();
    inline void processOthers();  // Small function, inline for performance
    inline void updateNextStep();  // Small function, inline for performance
    //!SECTION - per ctrl frequency

    void computeReward();
    void openDataFile();
    void writeData();
    
    // Joystick callback
    void joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);
    
    /////////////////////////////////// ONNX Runtime ///////////////////////////////////////
    // Actor network (main network)
    size_t input_number, output_number;
    std::vector<std::string> input_names, output_names;
    std::vector<const char *> input_names_char, output_names_char;
    std::vector<Ort::Value> input_tensors, output_tensors;
    std::vector<std::vector<float>> input_states_buffer;

    // Normalizer network
    size_t input_number_n, output_number_n;
    std::vector<std::string> input_names_n, output_names_n;
    std::vector<const char *> input_names_char_n, output_names_char_n;
    std::vector<Ort::Value> input_tensors_n, output_tensors_n;
    std::vector<std::vector<float>> input_states_buffer_n;

    // Decoder network
    size_t input_number_d, output_number_d;
    std::vector<std::string> input_names_d, output_names_d;
    std::vector<const char *> input_names_char_d, output_names_char_d;
    std::vector<Ort::Value> input_tensors_d, output_tensors_d;
    std::vector<std::vector<float>> input_states_buffer_d;

    // Encoder network
    size_t input_number_e, output_number_e;
    std::vector<std::string> input_names_e, output_names_e;
    std::vector<const char *> input_names_char_e, output_names_char_e;
    std::vector<Ort::Value> input_tensors_e, output_tensors_e;
    std::vector<std::vector<float>> input_states_buffer_e;

    // Critic network
    size_t input_number_c, output_number_c;
    std::vector<std::string> input_names_c, output_names_c;
    std::vector<const char *> input_names_char_c, output_names_char_c;
    std::vector<Ort::Value> input_tensors_c, output_tensors_c;
    std::vector<std::vector<float>> input_states_buffer_c;

    // Denormalizer network
    size_t input_number_dn, output_number_dn;
    std::vector<std::string> input_names_dn, output_names_dn;
    std::vector<const char *> input_names_char_dn, output_names_char_dn;
    std::vector<Ort::Value> input_tensors_dn, output_tensors_dn;
    std::vector<std::vector<float>> input_states_buffer_dn;

    // Optimization: Align vectors to cache line boundaries for better performance
    alignas(64) std::vector<float> state_cur_, critic_state_cur_, latent_cur_, h_cur_;
    alignas(64) std::vector<float> normalized_state_cur_, normalized_critic_state_cur_;

    //SECTION - temporary variables
    static constexpr int num_action = 12;
    static constexpr int num_actuator_action = 12;
    static constexpr int num_cur_state = 47;
    static constexpr int num_cur_critic_state = 149;
    static constexpr int num_cur_latent = 24;
    static constexpr int num_cur_h = 256;

    int input_obs_idx_ = 0;
    int input_h0_idx_ = 1;
    int output_action_idx_ = 0;
    int output_hn_idx_ = 1;
    int output_latent_idx_ = 2;

    Eigen::MatrixXd rl_action_;
    double value_ = 0.0;

    bool stop_by_value_thres_ = false;
    Eigen::Matrix<double, MODEL_DOF, 1> q_stop_;

    Eigen::Matrix<double, MODEL_DOF, 1> q_dot_lpf_;

    Eigen::Matrix<double, MODEL_DOF, 1> q_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_pre_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_vel_noise_;

    Eigen::Matrix<double, MODEL_DOF, 1> torque_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_spline_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_desired_spline_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_rl_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_desired_rl_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_bound_;
    Eigen::Matrix<double, num_action, 2> pd_limit;

    // Pre-computed PD limit values for optimization (avoids repeated calculations)
    Eigen::Matrix<double, num_action, 1> pd_limit_std_;   // (upper - lower) * 0.5
    Eigen::Matrix<double, num_action, 1> pd_limit_bias_;  // (upper + lower) * 0.5

    // Action scaling (matches training setup)
    Eigen::Matrix<double, num_action, 1> action_scale_;  // Action scale for position control

    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kp_;
    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kv_;

    float start_time_us = 0.0f;
    float time_inference_pre_us = 0.0f;

    double time_cur_s = 0.0;
    double time_pre_s = 0.0;

    Eigen::Vector3d base_lin_vel_, base_ang_vel_;
    double heading;
    
    // Walking/stepping related variables
    double step_period_ = 0.5;
    double step_ticks_ = 0.0;
    int phase_indicator_ = 0;
    
    // Command and control variables
    Eigen::Vector3d commands_;
    double target_heading_ = 0.0;
    bool heading_mode_ = false;
    
    // Joint bias
    Eigen::Matrix<double, MODEL_DOF, 1> q_bias_;
    
    // Robot configuration
    bool is_on_robot_ = false;
    
    // Joystick velocity scaling
    bool joy_enabled = false;
    double vel_scale_x_ = 0.3;
    double vel_scale_y_ = 0.1;
    
    // Time tracking
    double del_t = 0.0;

    // Task mode 6: Initial pose tracking variables
    Eigen::Matrix<double, MODEL_DOF, 1> q_init_target_;  // Target initial pose
    double init_pose_start_time_ = 0.0;  // Start time for trajectory
    double init_pose_duration_ = 3.0;    // Duration for smooth transition (seconds)
    bool init_pose_initialized_ = false; // Flag to track initialization
    //!SECTION - temporary variables

    // Configuration
    std::string config_file_path_;
    std::string data_output_dir_;
    std::string data_filename_prefix_;
    bool data_output_enabled_ = false;
    int save_frequency_ = 100;
    std::string ctrl_type_ = "T";  // Controller action type: "T" (torque) or "P" (position)

    // Debug configuration
    bool debug_enabled_ = true;  // Enable/disable debug output
    bool debug_enabled_torque = true;  // Enable/disable torque debug output
    int debug_print_frequency_ = 100;  // Print debug info every N iterations

private:
    //SECTION - temporary variables
    unsigned int walking_tick = 0;
    // RL controller local init flag (replaces non-existent dc_.tc_init)
    bool rl_tc_init_ = true;
    //!SECTION - temporary variables

    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::MemoryInfo> memory_info;
    std::unordered_map<std::string, std::unique_ptr<Ort::Session>> sessions_;
    
    // ROS2 subscribers
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    
    // Data logging
    std::ofstream writeFile;
    bool is_write_file_ = false;
};