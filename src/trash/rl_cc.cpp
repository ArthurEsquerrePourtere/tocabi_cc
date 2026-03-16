#include "rl_cc.hpp"
#include <algorithm>
#include <filesystem>
#include <map>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <yaml-cpp/yaml.h>

// ANSI Color Codes for Debug Output
#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_WHITE   "\033[37m"
#define COLOR_BOLD    "\033[1m"

// Optimized Debug Print Macros - only evaluate if debug enabled
#define DEBUG_HEADER(msg) if (debug_enabled_) { std::cout << COLOR_BOLD << COLOR_CYAN << "\n========== " << msg << " ==========" << COLOR_RESET << std::endl; }
#define DEBUG_SUCCESS(msg) if (debug_enabled_) { std::cout << COLOR_GREEN << "[✓] " << msg << COLOR_RESET << std::endl; }
#define DEBUG_INFO(msg) if (debug_enabled_) { std::cout << COLOR_BLUE << "[INFO] " << msg << COLOR_RESET << std::endl; }
#define DEBUG_WARN(msg) if (debug_enabled_) { std::cout << COLOR_YELLOW << "[WARN] " << msg << COLOR_RESET << std::endl; }
#define DEBUG_ERROR(msg) if (debug_enabled_) { std::cout << COLOR_RED << "[ERROR] " << msg << COLOR_RESET << std::endl; }
#define DEBUG_DATA(label, value) if (debug_enabled_) { std::cout << COLOR_MAGENTA << "[DATA] " << label << ": " << COLOR_WHITE << value << COLOR_RESET << std::endl; }

RlController::RlController(DataContainer &dc)
    : dc_(dc), dc_cc_(dc.node_)
{
    DEBUG_HEADER("INITIALIZING RL CONTROLLER");

    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "br_humanoid");
    memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    DEBUG_SUCCESS("ONNX Runtime environment created");

    auto& node = dc_.node_;
    node->declare_parameter<std::string>("config_file", "");
    config_file_path_ = node->get_parameter("config_file").as_string();

    if (config_file_path_.empty()) {
        #ifdef BR_RL_SOURCE_DIR
            config_file_path_ = std::string(BR_RL_SOURCE_DIR) + "/config/br_rl_config.yaml";
            DEBUG_INFO("Using source directory for config: " + config_file_path_);
        #else
            DEBUG_WARN("BR_RL_SOURCE_DIR not defined. Falling back to relative path.");
            RCLCPP_WARN(node->get_logger(), "BR_RL_SOURCE_DIR not defined. Falling back to relative path.");
            config_file_path_ = "./config/br_rl_config.yaml";
        #endif
    }

    DEBUG_INFO("Loading configuration from: " + config_file_path_);
    RCLCPP_INFO(node->get_logger(), "Loading configuration from: %s", config_file_path_.c_str());

    // Initialize variables and load configuration (single cohesive function)
    initVariablesAndLoadConfig(config_file_path_);

    if (joy_enabled){
        // Create joy subscriber
        joy_sub_ = node->create_subscription<sensor_msgs::msg::Joy>(
            "/joy_rui", 10,
            std::bind(&RlController::joyCallback, this, std::placeholders::_1));
            

        DEBUG_SUCCESS("Joy subscriber created on topic: /joy_rui");
        RCLCPP_INFO(node->get_logger(), "Joy subscriber created on topic: /joy_rui");
    }
    else {
        DEBUG_WARN("Joystick control disabled - not creating /joy_rui subscriber");
        RCLCPP_INFO(node->get_logger(), "Joystick control disabled - not creating /joy_rui subscriber");
        DEBUG_DATA("Static command mode - target_vel_x", commands_(0));
        DEBUG_DATA("Static command mode - target_vel_y", commands_(1));
        DEBUG_DATA("Static command mode - target_vel_yaw", commands_(2));
    }
}

RlController::~RlController()
{
    // Close data file if open
    if (writeFile.is_open()) {
        writeFile.close();
        RCLCPP_INFO(dc_.node_->get_logger(), "Data file closed");
    }
}

void RlController::initVariablesAndLoadConfig(const std::string& config_file_path)
{
    DEBUG_HEADER("INITIALIZING CONTROLLER");

    //? ===== STAGE 1: Initialize config-independent variables =====
    DEBUG_HEADER("Stage 1: Config-independent initialization");

    // Initialize action vector
    rl_action_.resize(num_action, 1);
    rl_action_.setZero();
    DEBUG_SUCCESS("Action vector initialized: " + std::to_string(num_action) + " dimensions");

    // Initialize state vectors for RL observation (using assign for better performance)
    state_cur_.assign(num_cur_state, 0.0f);
    critic_state_cur_.assign(num_cur_critic_state, 0.0f);
    normalized_state_cur_.assign(num_cur_state, 0.0f);
    normalized_critic_state_cur_.assign(num_cur_critic_state, 0.0f);
    h_cur_.assign(num_cur_h, 0.0f);
    latent_cur_.assign(num_cur_latent, 0.0f);
    DEBUG_SUCCESS("State vectors initialized");
    DEBUG_DATA("  - state_cur", num_cur_state);
    DEBUG_DATA("  - critic_state_cur", num_cur_critic_state);
    DEBUG_DATA("  - h_cur", num_cur_h);
    DEBUG_DATA("  - latent_cur", num_cur_latent);

    // Initialize robot state variables
    q_noise_.setZero();
    q_noise_pre_.setZero();
    q_vel_noise_.setZero();
    q_dot_lpf_.setZero();
    torque_init_.setZero();
    torque_spline_.setZero();
    q_desired_spline_.setZero();
    torque_rl_.setZero();
    q_desired_rl_.setZero();
    q_stop_.setZero();
    DEBUG_SUCCESS("Robot state variables initialized");

    // Initialize velocity tracking
    base_lin_vel_.setZero();
    base_ang_vel_.setZero();
    DEBUG_SUCCESS("Velocity tracking initialized");

    // Initialize walking/stepping variables
    step_period_ = 0.5;
    step_ticks_ = 0.0;
    phase_indicator_ = 0;
    DEBUG_DATA("Step period", step_period_);

    // Initialize command and control variables
    commands_.setZero();
    target_heading_ = 0.0;
    heading_mode_ = false;
    DEBUG_SUCCESS("Command variables initialized");

    // Initialize joint bias
    q_bias_.setZero();

    // Initialize joystick velocity scaling
    vel_scale_x_ = 0.3;
    vel_scale_y_ = 0.1;
    DEBUG_DATA("Velocity scales (x, y)", std::to_string(vel_scale_x_) + ", " + std::to_string(vel_scale_y_));

    // Default values (will be overridden by config file if present)
    q_init_.setZero();
    torque_bound_.setConstant(80.0);  // More efficient than setOnes() * 80.0
    DEBUG_DATA("Default torque bound", 80.0);

    // PD gains will be loaded from config
    kp_.setIdentity();
    kv_.setIdentity();
    DEBUG_SUCCESS("PD gains set to identity (will be overridden by config)");

    // pd_limit will be loaded from config
    pd_limit.setConstant(-1.0);  // Set all to -1.0
    pd_limit.col(1).setConstant(1.0);  // Set upper limits to 1.0
    DEBUG_SUCCESS("PD limits initialized to [-1.0, 1.0]");

    //? ===== STAGE 2: Load configuration and override defaults =====
    DEBUG_HEADER("Stage 2: Loading configuration file");

    try {
        const YAML::Node config = YAML::LoadFile(config_file_path);
        const auto& logger = dc_.node_->get_logger();

        // Get source directory for resolving relative paths
        std::string source_dir;
        #ifdef BR_RL_SOURCE_DIR
            source_dir = BR_RL_SOURCE_DIR;
            RCLCPP_INFO(logger, "Using source directory: %s", source_dir.c_str());
        #else
            RCLCPP_WARN(logger, "BR_RL_SOURCE_DIR not defined. Using current directory.");
            source_dir = ".";
        #endif

        // Load model paths
        std::map<std::string, std::string> model_paths;
        if (config["models"]) {
            for (const auto& model : config["models"]) {
                const std::string name = model.first.as<std::string>();
                std::string path = model.second.as<std::string>();

                // Resolve relative paths to absolute paths
                if (!path.empty() && path[0] != '/') {
                    // Check if it's just a filename (no directory separators)
                    if (path.find('/') == std::string::npos) {
                        // Assume it's in the onnx/ directory
                        path = source_dir + "/onnx/" + path;
                    } else {
                        // It's a relative path, prepend source directory
                        path = source_dir + "/" + path;
                    }
                }

                model_paths[name] = path;
                RCLCPP_INFO(logger, "Model '%s': %s", name.c_str(), path.c_str());
            }
        }

        // Load data output configuration
        if (config["data_output"]) {
            const auto& data_out = config["data_output"];
            data_output_enabled_ = data_out["enabled"].as<bool>(true);

            // Get output directory from config
            data_output_dir_ = data_out["output_dir"].as<std::string>("");

            // Process the output directory path
            if (data_output_dir_.empty()) {
                // Default: save to source_dir/data
                data_output_dir_ = source_dir + "/data";
            } else if (data_output_dir_[0] == '~') {
                // Expand ~ to home directory
                const char* home = std::getenv("HOME");
                if (home) {
                    data_output_dir_ = std::string(home) + data_output_dir_.substr(1);
                }
            } else if (data_output_dir_[0] != '/') {
                // Relative path: make it relative to source directory
                data_output_dir_ = source_dir + "/" + data_output_dir_;
            }
            // else: absolute path, use as-is

            data_filename_prefix_ = data_out["filename_prefix"].as<std::string>("rl_data");
            save_frequency_ = data_out["save_frequency"].as<int>(100);

            RCLCPP_INFO(logger, "Data output enabled: %s", data_output_enabled_ ? "true" : "false");
            RCLCPP_INFO(logger, "Data output directory: %s", data_output_dir_.c_str());

            if (data_output_enabled_) {
                std::filesystem::create_directories(data_output_dir_);
            }
        }

        if (config["controller"]) {
            ctrl_type_ = config["controller"]["action_type"].as<std::string>("P");
            is_on_robot_ = config["controller"]["is_on_robot"].as<bool>(false);

            // Load inference frequency
            if (config["controller"]["inference_frequency"]) {
                hz_ = config["controller"]["inference_frequency"].as<double>(125.0);
                RCLCPP_INFO(logger, "Inference frequency: %.1f Hz", hz_);
            }

            RCLCPP_INFO(logger, "Controller action type: %s", ctrl_type_.c_str());
            RCLCPP_INFO(logger, "Is on robot: %s", is_on_robot_ ? "true" : "false");
        }

        // Load controller parameters
        if (config["params"]) {
            const auto& params = config["params"];

            // Load torque bounds
            if (params["torque_bound"]) {
                auto torque_bounds = params["torque_bound"].as<std::vector<double>>();
                for (size_t i = 0; i < torque_bounds.size() && i < MODEL_DOF; ++i) {
                    torque_bound_(i) = torque_bounds[i];
                }
                RCLCPP_INFO(logger, "Loaded torque bounds for %zu joints", torque_bounds.size());
                std::stringstream ss;
                ss << "torque_bound: [";
                for (int i = 0; i < MODEL_DOF; ++i) {
                    ss << torque_bound_(i);
                    if (i < MODEL_DOF - 1) ss << ", ";
                }
                ss << "]";
                RCLCPP_INFO(logger, "%s", ss.str().c_str());
            }

            // Load kp gains
            if (params["kp_diagonal"] && params["kp_scale"]) {
                auto kp_values = params["kp_diagonal"].as<std::vector<double>>();
                double kp_scale = params["kp_scale"].as<double>(9.0);
                kp_.setZero();
                for (size_t i = 0; i < kp_values.size() && i < MODEL_DOF; ++i) {
                    kp_(i, i) = kp_values[i] * kp_scale;
                }
                RCLCPP_INFO(logger, "Loaded kp gains (scale: %.1f)", kp_scale);
                std::stringstream ss;
                ss << "kp diagonal: [";
                for (int i = 0; i < MODEL_DOF; ++i) {
                    ss << kp_(i, i);
                    if (i < MODEL_DOF - 1) ss << ", ";
                }
                ss << "]";
                RCLCPP_INFO(logger, "%s", ss.str().c_str());
            }

            // Load kv gains
            if (params["kv_diagonal"] && params["kv_scale"]) {
                auto kv_values = params["kv_diagonal"].as<std::vector<double>>();
                double kv_scale = params["kv_scale"].as<double>(3.0);
                kv_.setZero();
                for (size_t i = 0; i < kv_values.size() && i < MODEL_DOF; ++i) {
                    kv_(i, i) = kv_values[i] * kv_scale;
                }
                RCLCPP_INFO(logger, "Loaded kv gains (scale: %.1f)", kv_scale);
                std::stringstream ss;
                ss << "kv diagonal: [";
                for (int i = 0; i < MODEL_DOF; ++i) {
                    ss << kv_(i, i);
                    if (i < MODEL_DOF - 1) ss << ", ";
                }
                ss << "]";
                RCLCPP_INFO(logger, "%s", ss.str().c_str());
            }

            // Load q_init
            if (params["q_init"]) {
                auto q_init_values = params["q_init"].as<std::vector<double>>();
                for (size_t i = 0; i < q_init_values.size() && i < MODEL_DOF; ++i) {
                    q_init_(i) = q_init_values[i];
                }
                RCLCPP_INFO(logger, "Loaded initial joint positions (%zu values)", q_init_values.size());
                std::stringstream ss;
                ss << "q_init: [";
                for (int i = 0; i < MODEL_DOF; ++i) {
                    ss << q_init_(i);
                    if (i < MODEL_DOF - 1) ss << ", ";
                }
                ss << "]";
                RCLCPP_INFO(logger, "%s", ss.str().c_str());
            }

            // Load q_init_target for task_mode 5
            if (params["q_init_target"]) {
                auto q_init_target_values = params["q_init_target"].as<std::vector<double>>();
                for (size_t i = 0; i < q_init_target_values.size() && i < MODEL_DOF; ++i) {
                    q_init_target_(i) = q_init_target_values[i];
                }
                std::stringstream ss;
                ss << "q_init_target: [";
                for (int i = 0; i < MODEL_DOF; ++i) {
                    ss << q_init_target_(i);
                    if (i < MODEL_DOF - 1) ss << ", ";
                }
                ss << "]";
                RCLCPP_INFO(logger, "%s", ss.str().c_str());
    
                // Load init_pose_duration for task_mode 6
                if (params["init_pose_duration"]) {
                    init_pose_duration_ = params["init_pose_duration"].as<double>();
                    RCLCPP_INFO(logger, "Loaded init pose duration: %.2f seconds", init_pose_duration_);
                } else {
                    init_pose_duration_ = 3.0;  // Default 3 seconds
                    RCLCPP_WARN(logger, "No init_pose_duration in config, using default 3.0 seconds");
                }
            }

            // Load pd_limits and pre-compute values for optimization
            if (params["pd_limit"]) {
                auto pd_limits = params["pd_limit"].as<std::vector<std::vector<double>>>();
                for (size_t i = 0; i < pd_limits.size() && i < num_action; ++i) {
                    if (pd_limits[i].size() >= 2) {
                        pd_limit(i, 0) = pd_limits[i][0];  // Lower limit
                        pd_limit(i, 1) = pd_limits[i][1];  // Upper limit

                        // Pre-compute std and bias for fast PD control (avoids repeated calculation)
                        pd_limit_std_(i) = (pd_limit(i, 1) - pd_limit(i, 0)) * 0.5;
                        pd_limit_bias_(i) = (pd_limit(i, 1) + pd_limit(i, 0)) * 0.5;
                    }
                }
                RCLCPP_INFO(logger, "Loaded PD limits for %zu joints (pre-computed std/bias)", pd_limits.size());
            }

            // Load action_scale (matches training setup)
            if (params["action_scale"]) {
                auto action_scale_values = params["action_scale"].as<std::vector<double>>();
                for (size_t i = 0; i < action_scale_values.size() && i < num_action; ++i) {
                    action_scale_(i) = action_scale_values[i];
                }
                RCLCPP_INFO(logger, "Loaded action scale for %zu joints", action_scale_values.size());
                std::stringstream ss;
                ss << "action_scale: [";
                for (int i = 0; i < num_action; ++i) {
                    ss << action_scale_(i);
                    if (i < num_action - 1) ss << ", ";
                }
                ss << "]";
                RCLCPP_INFO(logger, "%s", ss.str().c_str());
            } else {
                // Default: all joints use 0.3 rad action scale (matching training)
                action_scale_.setConstant(0.3);
                RCLCPP_WARN(logger, "No action_scale in config, using default 0.3 for all joints");
            }
        }

        // Load command speed configuration
        if (config["commands"]) {
            const auto& commands = config["commands"];

            // Check if joystick is enabled
            joy_enabled = commands["joy_enabled"].as<bool>(false);
            RCLCPP_INFO(logger, "joy_enabled read from config: %s", joy_enabled ? "TRUE" : "FALSE");


            if (joy_enabled) {
                // Joystick mode: commands will be updated by joyCallback from /joy_rui topic
                RCLCPP_INFO(logger, "Joystick control enabled - listening to /joy_rui topic");
                
            } else {
                // Static command mode: use values from config file
                if (commands["target_vel_x"]) {
                    commands_(0) = commands["target_vel_x"].as<double>(0.0);
                }
                if (commands["target_vel_y"]) {
                    commands_(1) = commands["target_vel_y"].as<double>(0.0);
                }
                if (commands["target_vel_yaw"]) {
                    commands_(2) = commands["target_vel_yaw"].as<double>(0.0);
                }
                if (commands["heading"].as<bool>(false) && commands["target_heading"]) {
                    target_heading_ = commands["target_heading"].as<double>(0.0);
                }

                RCLCPP_INFO(logger, "Static command mode - using config file values");
                RCLCPP_INFO(logger, "target_vel_x: %.3f, target_vel_y: %.3f, target_vel_yaw: %.3f",
                            commands_(0), commands_(1), commands_(2));
                RCLCPP_INFO(logger, "target_heading: %.3f", target_heading_);
            }
        }

        // Load debug configuration
        if (config["debug"]) {
            const auto& debug = config["debug"];
            debug_enabled_ = debug["enable_debug_prints"].as<bool>(true);
            debug_enabled_torque = debug["enable_torque_debug_prints"].as<bool>(false);
            debug_print_frequency_ = debug["print_frequency"].as<int>(100);

            RCLCPP_INFO(logger, "Debug output enabled: %s", debug_enabled_ ? "true" : "false");
            RCLCPP_INFO(logger, "Torque debug output enabled: %s", debug_enabled_torque ? "true" : "false");
            RCLCPP_INFO(logger, "Debug print frequency: %d", debug_print_frequency_);
        }

        // Load ONNX models
        if (!model_paths.empty()) {
            loadOnnxModel(model_paths);
        } else {
            RCLCPP_WARN(logger, "No models specified in configuration file!");
        }

    } catch (const YAML::Exception& e) {
        RCLCPP_ERROR(dc_.node_->get_logger(), "Failed to load config file: %s", e.what());
        throw;
    }

    //? ===== STAGE 3: Finalize config-dependent initialization =====
    DEBUG_HEADER("Stage 3: Config-dependent initialization");

    // Initialize bias (depends on is_on_robot_ from config)
    initBias();

    // Calculate time step based on loaded inference frequency
    del_t = 1.0 / hz_;
    DEBUG_SUCCESS("Time step calculated from config");
    DEBUG_DATA("Inference frequency (hz_)", hz_);
    DEBUG_DATA("Time step (del_t)", del_t);
    RCLCPP_INFO(dc_.node_->get_logger(), "Inference frequency: %.1f Hz → time step: %.6f s", hz_, del_t);

    DEBUG_SUCCESS("Controller initialization complete");
}

void RlController::loadOnnxModel(const std::map<std::string, std::string>& model_paths)
{
    DEBUG_HEADER("LOADING ONNX MODELS");

    Ort::SessionOptions session_options;
    // Optimization: Enable all graph optimizations for better performance
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.AddConfigEntry("session.use_deterministic_compute", "1");
    // Optimization: Increase thread count for better CPU utilization
    session_options.SetIntraOpNumThreads(4);  // Increased from 2 for better parallelization
    session_options.SetInterOpNumThreads(1);
    // Optimization: Set execution mode to sequential for better cache locality
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    DEBUG_SUCCESS("ONNX session options configured");

    const auto& logger = dc_.node_->get_logger();

    // Load all models with error handling
    for (const auto& [name, path] : model_paths) {
        try {
            DEBUG_INFO("Loading model '" + name + "' from: " + path);
            RCLCPP_INFO(logger, "Loading %s model: %s", name.c_str(), path.c_str());
            sessions_[name] = std::make_unique<Ort::Session>(*env, path.c_str(), session_options);
            DEBUG_SUCCESS("Model '" + name + "' loaded successfully");
        } catch (const Ort::Exception& e) {
            RCLCPP_ERROR(logger, "Failed to load %s model: %s", name.c_str(), e.what());
            throw;
        }
    }

    Ort::AllocatorWithDefaultOptions allocator;

    //? ===== Actor Network =====
    // Get input/output info from actor network (main network)
    DEBUG_HEADER("CONFIGURING ACTOR NETWORK");
    auto* actor_session = getSession("actor");
    if (!actor_session) {
        DEBUG_ERROR("Actor session not found!");
        RCLCPP_ERROR(logger, "Actor session not found!");
        return;
    }
    DEBUG_SUCCESS("Actor session found");
    input_number = actor_session->GetInputCount();
    output_number = actor_session->GetOutputCount();
    DEBUG_DATA("Actor inputs", input_number);
    DEBUG_DATA("Actor outputs", output_number);

    input_names.resize(input_number);
    output_names.resize(output_number);
    input_names_char.resize(input_number);
    output_names_char.resize(output_number);

    // Get actor input names
    for (size_t i = 0; i < input_number; ++i) {
        Ort::AllocatedStringPtr input_name = actor_session->GetInputNameAllocated(i, allocator);
        input_names[i] = input_name.get();
        input_names_char[i] = input_names[i].c_str();
    }

    // Get actor output names
    for (size_t i = 0; i < output_number; ++i) {
        Ort::AllocatedStringPtr output_name = actor_session->GetOutputNameAllocated(i, allocator);
        output_names[i] = output_name.get();
        output_names_char[i] = output_names[i].c_str();
    }

    // Print actor input/output names with color
    std::cout << COLOR_BOLD << COLOR_GREEN << "Actor Input names: " << COLOR_WHITE;
    std::copy(input_names.begin(), input_names.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << COLOR_RESET << std::endl;

    std::cout << COLOR_BOLD << COLOR_GREEN << "Actor Output names: " << COLOR_WHITE;
    std::copy(output_names.begin(), output_names.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << COLOR_RESET << std::endl;

    // Dynamically find input indices by name
    for (size_t i = 0; i < input_number; ++i) {
        if (input_names[i] == "obs" || input_names[i] == "observations") {
            input_obs_idx_ = static_cast<int>(i);
            DEBUG_DATA("Found input 'obs' at index", input_obs_idx_);
        } else if (input_names[i] == "h0") {
            input_h0_idx_ = static_cast<int>(i);
            DEBUG_DATA("Found input 'h0' at index", input_h0_idx_);
        }
    }

    // Dynamically find output indices by name and shape
    // ONNX model outputs: action (12), hidden_state (256), prediction (149)
    // Note: Due to ONNX naming issues, we identify outputs by their shape
    for (size_t i = 0; i < output_number; ++i) {
        Ort::TypeInfo type_info = actor_session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_shape = tensor_info.GetShape();
        
        // Calculate last dimension (handles both [1, dim] and [1, 1, dim])
        int64_t last_dim = output_shape.back();
        
        if (output_names[i] == "action" || output_names[i] == "action_mean" || last_dim == num_action) {
            output_action_idx_ = static_cast<int>(i);
            DEBUG_DATA("Found output 'action' at index", output_action_idx_);
        } else if (last_dim == num_cur_h) {
            // Hidden state has dimension matching GRU hidden size (256)
            output_hn_idx_ = static_cast<int>(i);
            DEBUG_DATA("Found output 'hn' (hidden state, dim=" + std::to_string(last_dim) + ") at index", output_hn_idx_);
        } else if (last_dim == num_cur_critic_state) {
            // Prediction has dimension matching critic state (149)
            output_latent_idx_ = static_cast<int>(i);
            DEBUG_DATA("Found output 'prediction' (dim=" + std::to_string(last_dim) + ") at index", output_latent_idx_);
        }
    }
    
    // Validate that all indices were found
    if (output_action_idx_ < 0 || output_hn_idx_ < 0) {
        RCLCPP_ERROR(logger, "Failed to find required output indices! action_idx=%d, hn_idx=%d", 
                     output_action_idx_, output_hn_idx_);
    }

    // Initialize actor input tensors
    DEBUG_INFO("Initializing actor input tensors...");
    for (size_t i = 0; i < input_number; ++i) {
        Ort::TypeInfo type_info = actor_session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = tensor_info.GetShape();

        std::cout << COLOR_MAGENTA << "  Actor Input " << i << " (" << input_names[i] << ") shape: " << COLOR_WHITE;
        for (const auto& dim : input_shape) {
            std::cout << dim << " ";
        }
        std::cout << COLOR_RESET << std::endl;

        size_t element_count = computeElementCount(input_shape);
        if (element_count == 0) {
            RCLCPP_ERROR(logger, "Cannot determine element count for dynamic shape in actor input %zu", i);
            return;
        }
        std::vector<float> input_tensor_values(element_count, 0.0f);
        input_states_buffer.push_back(std::move(input_tensor_values));

        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            *memory_info,
            input_states_buffer.back().data(),
            input_states_buffer.back().size(),
            input_shape.data(),
            input_shape.size()));
    }
    DEBUG_SUCCESS("Actor input tensors initialized");

    //? ===== Critic Network (if exists) =====
    DEBUG_HEADER("CONFIGURING CRITIC NETWORK");
    auto* critic_session = getSession("critic");
    if (critic_session) {
        DEBUG_SUCCESS("Critic session found");
        input_number_c = critic_session->GetInputCount();
        output_number_c = critic_session->GetOutputCount();
        DEBUG_DATA("Critic inputs", input_number_c);
        DEBUG_DATA("Critic outputs", output_number_c);

        input_names_c.resize(input_number_c);
        output_names_c.resize(output_number_c);
        input_names_char_c.resize(input_number_c);
        output_names_char_c.resize(output_number_c);

        for (size_t i = 0; i < input_number_c; ++i) {
            Ort::AllocatedStringPtr input_name_c = critic_session->GetInputNameAllocated(i, allocator);
            input_names_c[i] = input_name_c.get();
            input_names_char_c[i] = input_names_c[i].c_str();
        }

        for (size_t i = 0; i < output_number_c; ++i) {
            Ort::AllocatedStringPtr output_name_c = critic_session->GetOutputNameAllocated(i, allocator);
            output_names_c[i] = output_name_c.get();
            output_names_char_c[i] = output_names_c[i].c_str();
        }

        std::cout << COLOR_BOLD << COLOR_GREEN << "Critic Input names: " << COLOR_WHITE;
        std::copy(input_names_c.begin(), input_names_c.end(), std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << COLOR_RESET << std::endl;

        std::cout << COLOR_BOLD << COLOR_GREEN << "Critic Output names: " << COLOR_WHITE;
        std::copy(output_names_c.begin(), output_names_c.end(), std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << COLOR_RESET << std::endl;

        // Initialize critic input tensors
        DEBUG_INFO("Initializing critic input tensors...");
        for (size_t i = 0; i < input_number_c; ++i) {
            Ort::TypeInfo type_info = critic_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_shape_c = tensor_info.GetShape();

            std::cout << COLOR_MAGENTA << "  Critic Input " << i << " shape: " << COLOR_WHITE;
            for (const auto& dim : input_shape_c) {
                std::cout << dim << " ";
            }
            std::cout << COLOR_RESET << std::endl;

            size_t element_count = computeElementCount(input_shape_c);
            if (element_count == 0) {
                RCLCPP_ERROR(logger, "Cannot determine element count for dynamic shape in critic input %zu", i);
                return;
            }
            std::vector<float> input_tensor_values(element_count, 0.0f);
            input_states_buffer_c.push_back(std::move(input_tensor_values));

            input_tensors_c.emplace_back(Ort::Value::CreateTensor<float>(
                *memory_info,
                input_states_buffer_c.back().data(),
                input_states_buffer_c.back().size(),
                input_shape_c.data(),
                input_shape_c.size()));
        }
        DEBUG_SUCCESS("Critic input tensors initialized");
    } else {
        DEBUG_WARN("No critic session found");
    }

    //? ===== Decoder Network (if exists) =====
    DEBUG_HEADER("CONFIGURING DECODER NETWORK");
    auto* decoder_session = getSession("decoder");
    if (decoder_session) {
        DEBUG_SUCCESS("Decoder session found");
        input_number_d = decoder_session->GetInputCount();
        output_number_d = decoder_session->GetOutputCount();
        DEBUG_DATA("Decoder inputs", input_number_d);
        DEBUG_DATA("Decoder outputs", output_number_d);
        
        input_names_d.resize(input_number_d);
        output_names_d.resize(output_number_d);
        input_names_char_d.resize(input_number_d);
        output_names_char_d.resize(output_number_d);
        
        for (size_t i = 0; i < input_number_d; ++i) {
            Ort::AllocatedStringPtr input_name_d = decoder_session->GetInputNameAllocated(i, allocator);
            input_names_d[i] = input_name_d.get();
            input_names_char_d[i] = input_names_d[i].c_str();
        }
        
        for (size_t i = 0; i < output_number_d; ++i) {
            Ort::AllocatedStringPtr output_name_d = decoder_session->GetOutputNameAllocated(i, allocator);
            output_names_d[i] = output_name_d.get();
            output_names_char_d[i] = output_names_d[i].c_str();
        }
        
        std::cout << COLOR_BOLD << COLOR_GREEN << "Decoder Input names: " << COLOR_WHITE;
        std::copy(input_names_d.begin(), input_names_d.end(), std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << COLOR_RESET << std::endl;
        
        std::cout << COLOR_BOLD << COLOR_GREEN << "Decoder Output names: " << COLOR_WHITE;
        std::copy(output_names_d.begin(), output_names_d.end(), std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << COLOR_RESET << std::endl;
        
        // Initialize decoder input tensors
        DEBUG_INFO("Initializing decoder input tensors...");
        for (size_t i = 0; i < input_number_d; ++i) {
            Ort::TypeInfo type_info = decoder_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_shape_d = tensor_info.GetShape();
            
            std::cout << COLOR_MAGENTA << "  Decoder Input " << i << " shape: " << COLOR_WHITE;
            for (const auto& dim : input_shape_d) {
                std::cout << dim << " ";
            }
            std::cout << COLOR_RESET << std::endl;

            size_t element_count = computeElementCount(input_shape_d);
            if (element_count == 0) {
                RCLCPP_ERROR(logger, "Cannot determine element count for dynamic shape in decoder input %zu", i);
                return;
            }
            std::vector<float> input_tensor_values(element_count, 0.0f);
            input_states_buffer_d.push_back(std::move(input_tensor_values));
            input_tensors_d.emplace_back(Ort::Value::CreateTensor<float>(
                *memory_info,
                input_states_buffer_d.back().data(),
                input_states_buffer_d.back().size(),
                input_shape_d.data(),
                input_shape_d.size()));
            }
            DEBUG_SUCCESS("Decoder input tensors initialized");
    } else {
        DEBUG_WARN("No decoder session found");
    }

    //? ===== Denormalizer Network (if exists) =====
    DEBUG_HEADER("CONFIGURING DENORMALIZER NETWORK");
    auto* denormalizer_session = getSession("denormalizer");
    if (denormalizer_session) {
        DEBUG_SUCCESS("Denormalizer session found");
        input_number_dn = denormalizer_session->GetInputCount();
        output_number_dn = denormalizer_session->GetOutputCount();
        DEBUG_DATA("Denormalizer inputs", input_number_dn);
        DEBUG_DATA("Denormalizer outputs", output_number_dn);

        input_names_dn.resize(input_number_dn);
        output_names_dn.resize(output_number_dn);
        input_names_char_dn.resize(input_number_dn);
        output_names_char_dn.resize(output_number_dn);

        for (size_t i = 0; i < input_number_dn; ++i) {
            Ort::AllocatedStringPtr input_name_dn = denormalizer_session->GetInputNameAllocated(i, allocator);
            input_names_dn[i] = input_name_dn.get();
            input_names_char_dn[i] = input_names_dn[i].c_str();
        }

        for (size_t i = 0; i < output_number_dn; ++i) {
            Ort::AllocatedStringPtr output_name_dn = denormalizer_session->GetOutputNameAllocated(i, allocator);
            output_names_dn[i] = output_name_dn.get();
            output_names_char_dn[i] = output_names_dn[i].c_str();
        }

        std::cout << COLOR_BOLD << COLOR_GREEN << "Denormalizer Input names: " << COLOR_WHITE;
        std::copy(input_names_dn.begin(), input_names_dn.end(), std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << COLOR_RESET << std::endl;

        std::cout << COLOR_BOLD << COLOR_GREEN << "Denormalizer Output names: " << COLOR_WHITE;
        std::copy(output_names_dn.begin(), output_names_dn.end(), std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << COLOR_RESET << std::endl;

        // Initialize denormalizer input tensors
        DEBUG_INFO("Initializing denormalizer input tensors...");
        for (size_t i = 0; i < input_number_dn; ++i) {
            Ort::TypeInfo type_info = denormalizer_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_shape_dn = tensor_info.GetShape();

            std::cout << COLOR_MAGENTA << "  Denormalizer Input " << i << " shape: " << COLOR_WHITE;
            for (const auto& dim : input_shape_dn) {
                std::cout << dim << " ";
            }
            std::cout << COLOR_RESET << std::endl;

            size_t element_count = computeElementCount(input_shape_dn);
            if (element_count == 0) {
                RCLCPP_ERROR(logger, "Cannot determine element count for dynamic shape in denormalizer input %zu", i);
                return;
            }
            std::vector<float> input_tensor_values(element_count, 0.0f);
            input_states_buffer_dn.push_back(std::move(input_tensor_values));
            input_tensors_dn.emplace_back(Ort::Value::CreateTensor<float>(
                *memory_info,
                input_states_buffer_dn.back().data(),
                input_states_buffer_dn.back().size(),
                input_shape_dn.data(),
                input_shape_dn.size()));
        }
        DEBUG_SUCCESS("Denormalizer input tensors initialized");
    } else {
        DEBUG_WARN("No denormalizer session found - using raw observations");
    }

    //? ===== Encoder Network (if exists) =====
    DEBUG_HEADER("CONFIGURING ENCODER NETWORK");
    auto* encoder_session = getSession("encoder");
    if (encoder_session) {
        DEBUG_SUCCESS("Encoder session found");
        input_number_e = encoder_session->GetInputCount();
        output_number_e = encoder_session->GetOutputCount();
        DEBUG_DATA("Encoder inputs", input_number_e);
        DEBUG_DATA("Encoder outputs", output_number_e);
        
        input_names_e.resize(input_number_e);
        output_names_e.resize(output_number_e);
        input_names_char_e.resize(input_number_e);
        output_names_char_e.resize(output_number_e);
        
        for (size_t i = 0; i < input_number_e; ++i) {
            Ort::AllocatedStringPtr input_name_e = encoder_session->GetInputNameAllocated(i, allocator);
            input_names_e[i] = input_name_e.get();
            input_names_char_e[i] = input_names_e[i].c_str();
        }
        
        for (size_t i = 0; i < output_number_e; ++i) {
            Ort::AllocatedStringPtr output_name_e = encoder_session->GetOutputNameAllocated(i, allocator);
            output_names_e[i] = output_name_e.get();
            output_names_char_e[i] = output_names_e[i].c_str();
        }
        
        std::cout << COLOR_BOLD << COLOR_GREEN << "Encoder Input names: " << COLOR_WHITE;
        std::copy(input_names_e.begin(), input_names_e.end(), std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << COLOR_RESET << std::endl;
        
        std::cout << COLOR_BOLD << COLOR_GREEN << "Encoder Output names: " << COLOR_WHITE;
        std::copy(output_names_e.begin(), output_names_e.end(), std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << COLOR_RESET << std::endl;
        
        // Initialize encoder input tensors
        DEBUG_INFO("Initializing encoder input tensors...");
        for (size_t i = 0; i < input_number_e; ++i) {
            Ort::TypeInfo type_info = encoder_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_shape_e = tensor_info.GetShape();
            
            std::cout << COLOR_MAGENTA << "  Encoder Input " << i << " shape: " << COLOR_WHITE;
            for (const auto& dim : input_shape_e) {
                std::cout << dim << " ";
            }
            std::cout << COLOR_RESET << std::endl;

            size_t element_count = computeElementCount(input_shape_e);
            if (element_count == 0) {
                RCLCPP_ERROR(logger, "Cannot determine element count for dynamic shape in encoder input %zu", i);
                return;
            }
            std::vector<float> input_tensor_values(element_count, 0.0f);
            input_states_buffer_e.push_back(std::move(input_tensor_values));
            input_tensors_e.emplace_back(Ort::Value::CreateTensor<float>(
                *memory_info,
                input_states_buffer_e.back().data(),
                input_states_buffer_e.back().size(),
                input_shape_e.data(),
                input_shape_e.size()));
            }
            DEBUG_SUCCESS("Encoder input tensors initialized");
    } else {
        DEBUG_WARN("No encoder session found");
    }

    //? ===== Normalizer Network (if exists) =====
    DEBUG_HEADER("CONFIGURING NORMALIZER NETWORK");
    auto* normalizer_session = getSession("normalizer");
    if (normalizer_session) {
        DEBUG_SUCCESS("Normalizer session found");
        input_number_n = normalizer_session->GetInputCount();
        output_number_n = normalizer_session->GetOutputCount();
        DEBUG_DATA("Normalizer inputs", input_number_n);
        DEBUG_DATA("Normalizer outputs", output_number_n);

        input_names_n.resize(input_number_n);
        output_names_n.resize(output_number_n);
        input_names_char_n.resize(input_number_n);
        output_names_char_n.resize(output_number_n);

        for (size_t i = 0; i < input_number_n; ++i) {
            Ort::AllocatedStringPtr input_name_n = normalizer_session->GetInputNameAllocated(i, allocator);
            input_names_n[i] = input_name_n.get();
            input_names_char_n[i] = input_names_n[i].c_str();
        }

        for (size_t i = 0; i < output_number_n; ++i) {
            Ort::AllocatedStringPtr output_name_n = normalizer_session->GetOutputNameAllocated(i, allocator);
            output_names_n[i] = output_name_n.get();
            output_names_char_n[i] = output_names_n[i].c_str();
        }

        std::cout << COLOR_BOLD << COLOR_GREEN << "Normalizer Input names: " << COLOR_WHITE;
        std::copy(input_names_n.begin(), input_names_n.end(), std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << COLOR_RESET << std::endl;

        std::cout << COLOR_BOLD << COLOR_GREEN << "Normalizer Output names: " << COLOR_WHITE;
        std::copy(output_names_n.begin(), output_names_n.end(), std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << COLOR_RESET << std::endl;

        // Initialize normalizer input tensors
        DEBUG_INFO("Initializing normalizer input tensors...");
        for (size_t i = 0; i < input_number_n; ++i) {
            Ort::TypeInfo type_info = normalizer_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_shape_n = tensor_info.GetShape();

            std::cout << COLOR_MAGENTA << "  Normalizer Input " << i << " shape: " << COLOR_WHITE;
            for (const auto& dim : input_shape_n) {
                std::cout << dim << " ";
            }
            std::cout << COLOR_RESET << std::endl;

            size_t element_count = computeElementCount(input_shape_n);
            if (element_count == 0) {
                RCLCPP_ERROR(logger, "Cannot determine element count for dynamic shape in normalizer input %zu", i);
                return;
            }
            std::vector<float> input_tensor_values(element_count, 0.0f);
            input_states_buffer_n.push_back(std::move(input_tensor_values));

            input_tensors_n.emplace_back(Ort::Value::CreateTensor<float>(
                *memory_info,
                input_states_buffer_n.back().data(),
                input_states_buffer_n.back().size(),
                input_shape_n.data(),
                input_shape_n.size()));
        }
        DEBUG_SUCCESS("Normalizer input tensors initialized");
    } else {
        DEBUG_WARN("No normalizer session found - using raw observations");
    }

    DEBUG_HEADER("MODEL LOADING COMPLETE");
    std::cout << COLOR_BOLD << COLOR_GREEN << "✓ All models loaded successfully!" << COLOR_RESET << std::endl;
    DEBUG_DATA("Actor", std::to_string(input_number) + " inputs, " + std::to_string(output_number) + " outputs");
    RCLCPP_INFO(logger, "ONNX models loaded successfully. Actor: %zu inputs, %zu outputs", input_number, output_number);
}

Ort::Session* RlController::getSession(const std::string& name)
{
    const auto it = sessions_.find(name);
    return (it != sessions_.end()) ? it->second.get() : nullptr;
}

size_t RlController::computeElementCount(const std::vector<int64_t>& shape)
{
    size_t count = 1;
    for (const auto& dim : shape) {
        if (dim <= 0) {
            // Dynamic dimension (like batch size -1), use 1 as default
            DEBUG_WARN("Dynamic dimension detected in shape, using 1 as placeholder for batch dimension");
            count *= 1;  // Assume batch size of 1 for dynamic dimensions
        } else {
            count *= static_cast<size_t>(dim);
        }
    }
    return count;
}

void RlController::computeFast() // update 1ms
{
    copyRobotData(dc_);
    
    // TODO: add RL control (ex. compute obs, action, etc.)

    // Task mode 5: Smooth trajectory to initial pose with cubic interpolation
    if (dc_cc_.task_cmd_.task_mode == 5)
    {
        // Get current joint positions
        Eigen::Matrix<double, MODEL_DOF, 1> q_current = dc_cc_.rd_.q_virtual_.segment(7, MODEL_DOF);

        // Static variable to store the starting pose for trajectory
        static Eigen::Matrix<double, MODEL_DOF, 1> q_start;

        // Initialize trajectory on first entry
        if (!init_pose_initialized_) {
            init_pose_start_time_ = dc_cc_.rd_.control_time_;
            init_pose_initialized_ = true;
            q_start = q_current;  // Capture current pose as starting pose
            DEBUG_HEADER("INITIALIZING SMOOTH TRAJECTORY TO INIT POSE");

        }

        // Current time
        double time_current = dc_cc_.rd_.control_time_;
        double time_end = init_pose_start_time_ + init_pose_duration_;

        // Generate desired position using cubic trajectory
        Eigen::Matrix<double, MODEL_DOF, 1> q_desired = DyrosMath::cubicVector<MODEL_DOF>(
            time_current,
            init_pose_start_time_,
            time_end,
            q_start,
            q_init_target_,
            Eigen::Matrix<double, MODEL_DOF, 1>::Zero(),  // Start with zero velocity
            Eigen::Matrix<double, MODEL_DOF, 1>::Zero()   // End with zero velocity
        );

        // Check if trajectory is complete
        bool trajectory_complete = (time_current >= time_end);

        // Calculate error for monitoring
        Eigen::Matrix<double, MODEL_DOF, 1> pose_error = (q_current - q_init_target_).cwiseAbs();
        double max_error = pose_error.maxCoeff();
        int max_error_idx = 0;
        pose_error.maxCoeff(&max_error_idx);

        if (trajectory_complete)
        {
            const double pose_tolerance = 0.005;  // ~0.057 degrees
            bool is_in_init_pose = (max_error < pose_tolerance);

            if (is_in_init_pose) {
                RCLCPP_INFO_THROTTLE(dc_.node_->get_logger(), *dc_.node_->get_clock(), 1000,
                    "✓ Trajectory complete! Robot at initial pose (max error: %.4f rad at joint %d)",
                    max_error, max_error_idx);
            } else {
                RCLCPP_WARN_THROTTLE(dc_.node_->get_logger(), *dc_.node_->get_clock(), 1000,
                    "Trajectory complete but error remains: %.4f rad (%.2f deg) at joint %d | Current: %.4f, Target: %.4f, Desired: %.4f",
                    max_error, max_error * 180.0 / M_PI, max_error_idx,
                    q_current(max_error_idx), q_init_target_(max_error_idx), q_desired(max_error_idx));
            }
        } else {
            double progress = (time_current - init_pose_start_time_) / init_pose_duration_ * 100.0;
            RCLCPP_INFO_THROTTLE(dc_.node_->get_logger(), *dc_.node_->get_clock(), 1000,
                "Moving to init pose: %.1f%% complete (max error: %.4f rad at joint %d)",
                progress, max_error, max_error_idx);
        }

        DEBUG_DATA("Desired Position", q_desired.transpose());
        DEBUG_DATA("Current Position", q_current.transpose());

        // Apply PD control to track desired trajectory
        for (int i = 0; i < MODEL_DOF; ++i) {
            dc_.rd_.torque_desired(i) = kp_(i,i) * (q_desired(i) - q_current(i))
                                      - kv_(i,i) * dc_cc_.rd_.q_dot_virtual_(7 + i);
        }

    }

    if (dc_cc_.task_cmd_.task_mode == 6)  // RL Control Mode
    {
        if (rl_tc_init_)
        {   
            DEBUG_WARN("task_mode == 6: RL Control Loop Initialization");
            DEBUG_HEADER("INITIALIZING RL CONTROL LOOP");
            start_time_us = dc_cc_.rd_.control_time_ * 1e6;  // Convert to microseconds once
    
            q_noise_pre_ = q_noise_ = q_init_ = dc_cc_.rd_.q_virtual_.segment(7,MODEL_DOF);
            
            time_cur_s = start_time_us * 1.0e-6;
            time_pre_s = time_cur_s - 0.005;
    
            time_inference_pre_us = start_time_us - (del_t)*1e6;
            rl_tc_init_ = false;
    
            DEBUG_DATA("Current time", time_cur_s);
            DEBUG_DATA("Inference frequency", hz_);
    
            torque_init_ = dc_cc_.rd_.torque_desired;
            DEBUG_SUCCESS("Initial torque saved");
    
            // Initialize data logging if enabled
            if (data_output_enabled_) {
                DEBUG_INFO("Opening data file for logging...");
                openDataFile();
            }
    
            DEBUG_INFO("Processing initial observation...");
            processNoise();
            processBias();
            processObservation();
            DEBUG_SUCCESS("RL Controller initialized successfully!");
        }
    
        processNoise();
        processBias();
        
        // Check for NaN in noise data (defensive check)
        validateAndFixNaN(q_noise_, q_init_);
        validateAndFixNaN(q_vel_noise_);
        
        if ((dc_cc_.rd_.control_time_ * 1.0e6 - time_inference_pre_us)*1.0e-6 >= del_t)
        {
            processObservation();
            feedforwardPolicy();
            processOthers();  // ✓ MOVED HERE - Update hidden state immediately after network inference
            updateNextStep();
    
            if (value_ < 0.0 && !stop_by_value_thres_)
            {
                stop_by_value_thres_ = true;
                q_stop_ = q_noise_;
                RCLCPP_ERROR(dc_.node_->get_logger(), "Stop by Value Function at tick %u, Value: %.3f", walking_tick, value_);
            }
        }
    
        // Compute torques based on control type - cache mode check once
        const bool is_torque_mode = (ctrl_type_ == "T");

        // SMOOTH STARTUP: Scale actions during initial ramp period to prevent jerky motion
        constexpr double ramp_duration_us = 0.5e6;  // 0.5 seconds ramp time
        double action_scale_factor = 1.0;
        if (dc_cc_.rd_.control_time_ * 1.0e6 < start_time_us + ramp_duration_us) {
            // Cubic ramp from 0 to 1 over ramp duration
            double elapsed_us = dc_cc_.rd_.control_time_ * 1.0e6 - start_time_us;
            action_scale_factor = DyrosMath::cubic(elapsed_us, 0.0, ramp_duration_us, 0.0, 1.0, 0.0, 0.0);
        }

        // Optimization: Use std::clamp for better readability and potential vectorization
        constexpr double action_min = -1.0, action_max = 1.0;
        // cout << "torque_bound_: " << torque_bound_.transpose() << endl;
        if (is_torque_mode) {
            // Torque mode: simple scaling with startup ramp
            for (int i = 0; i < num_actuator_action; ++i) {
                const double action_clamped = std::clamp(rl_action_(i), action_min, action_max);
                torque_rl_(i) = action_clamped * torque_bound_(i) * action_scale_factor;
            }
        } else {
            // Position control mode: PD control MATCHING TRAINING SETUP with startup ramp
            // Training formula: torque = Kp * (action * action_scale + default_pos - current_pos) - Kd * vel
            // Optimization: Pre-fetch pointers for better cache performance
            const double* q_init_data = q_init_.data();
            const double* q_noise_data = q_noise_.data();
            const double* q_vel_data = q_vel_noise_.data();
            const double* action_scale_data = action_scale_.data();
            double* torque_data = torque_rl_.data();

            for (int i = 0; i < num_actuator_action; ++i) {
                const double action_clamped = std::clamp(rl_action_(i), action_min, action_max);
                // Optimization: Cache diagonal elements to avoid repeated matrix access
                const double kp = kp_(i,i);
                const double kv = kv_(i,i);
                // Apply action_scale_factor to smoothly ramp up the action contribution
                // torque_data[i] = kp * (action_clamped * action_scale_data[i] * action_scale_factor + q_init_data[i] - q_noise_data[i]) - kv * q_vel_data[i];
                q_desired_rl_(i) = action_clamped * action_scale_data[i] + q_init_data[i];

            }
        }
    
        // Handle non-actuated joints (optimized PD control)
        for (int i = num_actuator_action; i < MODEL_DOF; ++i) {
            // torque_rl_(i) = kp_(i,i) * (q_init_(i) - q_noise_(i)) - kv_(i,i) * q_vel_noise_(i);
            q_desired_rl_(i) = q_init_(i);
        }
    
        // Debug output: Print computed torques for leg joints
        static int torque_call_count = 0;
        torque_call_count++;
        const bool should_debug_torque = debug_enabled_torque && debug_enabled_ && (torque_call_count % debug_print_frequency_ == 1);
    
        if (should_debug_torque) {
            std::cout << COLOR_BOLD << COLOR_GREEN << "\n========== TORQUE COMPUTATION #" << torque_call_count << " ==========" << COLOR_RESET << std::endl;
            std::cout << COLOR_GREEN << "[TORQUE MODE] " << (is_torque_mode ? "Direct torque (T)" : "Position control (P)") << COLOR_RESET << std::endl;
    
            // Print leg torques with joint names
            const char* leg_joint_names[] = {"L_HipRoll", "L_HipPitch", "L_HipYaw", "L_Knee", "L_AnklePitch", "L_AnkleRoll",
                                            "R_HipRoll", "R_HipPitch", "R_HipYaw", "R_Knee", "R_AnklePitch", "R_AnkleRoll"};
    
            std::cout << COLOR_GREEN << "[TORQUES] Leg joint torques (N·m):" << COLOR_RESET << std::endl;
            for (int i = 0; i < std::min(12, num_actuator_action); ++i) {
                std::cout << "  " << leg_joint_names[i] << ": " << std::fixed << std::setprecision(2) << torque_rl_(i) << " N·m";
                if (is_torque_mode) {
                    std::cout << " (action: " << std::setprecision(4) << DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0) << " × bound: " << torque_bound_(i) << ")";
                }
                std::cout << std::endl;
            }
    
            // Print torque statistics
            double torque_mean = 0.0, torque_max = -1e9, torque_min = 1e9;
            for (int i = 0; i < std::min(12, num_actuator_action); ++i) {
                torque_mean += std::abs(torque_rl_(i));
                torque_max = std::max(torque_max, std::abs(torque_rl_(i)));
                torque_min = std::min(torque_min, std::abs(torque_rl_(i)));
            }
            torque_mean /= std::min(12, num_actuator_action);
            std::cout << COLOR_YELLOW << "[TORQUE STATS] Mean(abs): " << std::fixed << std::setprecision(2) << torque_mean << " N·m" << " | Max(abs): " << torque_max << " N·m | Min(abs): " << torque_min << " N·m" << COLOR_RESET << std::endl;
    
            // Diagnostic: Check if torques are suspiciously weak
            if (torque_mean < 20.0) {
                std::cout << COLOR_BOLD << COLOR_RED << "⚠️  WARNING: Average torque is very low (" << torque_mean << " N·m)!" << COLOR_RESET << std::endl;
                std::cout << COLOR_RED << "    Possible causes:" << COLOR_RESET << std::endl;
                std::cout << COLOR_RED << "    1. Network actions are too small (check [ACTION STATS] above)" << COLOR_RESET << std::endl;
                std::cout << COLOR_RED << "    2. Wrong ONNX model loaded (check model timestamp)" << COLOR_RESET << std::endl;
                std::cout << COLOR_RED << "    3. Observation normalization mismatch" << COLOR_RESET << std::endl;
            }
        }
    
        // Apply torque with ramp-up or emergency stop
        if (stop_by_value_thres_) {
            dc_.rd_.torque_desired = kp_ * (q_stop_ - q_noise_) - kv_ * q_vel_noise_;
            RCLCPP_ERROR(dc_.node_->get_logger(), "Emergency Stop Position Control Applied!");
        } else if (dc_cc_.rd_.control_time_ * 1.0e6 < start_time_us + 0.5e6) {  // Increased from 0.3s to 2.0s for smoother startup
            const double ramp_end_time_us = start_time_us + 0.5e6;
            for (int i = 0; i < MODEL_DOF; ++i) {
                // torque_spline_(i) = DyrosMath::cubic(dc_cc_.rd_.control_time_ * 1.0e6, start_time_us, ramp_end_time_us, torque_init_(i), torque_rl_(i), 0.0, 0.0);
                q_desired_spline_(i) = DyrosMath::cubic(dc_cc_.rd_.control_time_ * 1.0e6, start_time_us, ramp_end_time_us, q_init_(i), q_desired_rl_(i), 0.0, 0.0);
            }
            // dc_.rd_.torque_desired = torque_spline_;
            dc_.rd_.q_desired = q_desired_spline_;
        } else {
            // dc_.rd_.torque_desired = torque_rl_;
            dc_.rd_.q_desired = q_desired_rl_;

        }
        // dc_.rd_.torque_desired = torque_rl_;
    }
}

void RlController::computeSlow() // update 10ms
{
    // Dont use this for general case
}

void RlController::computeMpc() // update 10ms
{
    // Dont use this for general case
}

void RlController::copyRobotData(DataContainer &dc_global_)
{
    memcpy(&dc_cc_, &dc_global_, sizeof(DataContainer));
}

void RlController::processNoise()
{
    time_cur_s = dc_cc_.rd_.control_time_;  // Keep in seconds for time calculations
    const double time_delta = time_cur_s - time_pre_s;
    const bool valid_time_delta = (time_delta > 0.0 && time_delta < 1.0);  // Sanity check: time_delta should be < 1 second

    if (is_on_robot_)
    {
        // On real robot: use actual sensor data - use direct assignment instead of segment copy
        q_vel_noise_.noalias() = dc_cc_.rd_.q_dot_virtual_.segment(6, MODEL_DOF);
        q_noise_.noalias() = dc_cc_.rd_.q_virtual_.segment(7, MODEL_DOF);

        if (valid_time_delta)
        {
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1.0 / time_delta, 4.0);
        }
    }
    else
    {
        // In simulation: add noise to joint positions
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(-0.00001, 0.00001);

        // Vectorized noise addition
        q_noise_.noalias() = dc_cc_.rd_.q_virtual_.segment(7, MODEL_DOF);
        for (int i = 0; i < MODEL_DOF; i++) {
            q_noise_(i) += dis(gen);
        }

        if (valid_time_delta)
        {
            q_vel_noise_.noalias() = (q_noise_ - q_noise_pre_) / time_delta;
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1.0 / time_delta, 4.0);
        }
        // else: No need to reassign - values remain unchanged

        q_noise_pre_ = q_noise_;
    }

    time_pre_s = time_cur_s;
}

void RlController::initBias()
{
    q_bias_.setZero();
    if (!is_on_robot_) {  // Fixed: was ~is_on_robot_ (bitwise NOT)
        std::random_device rd;
        std::mt19937 gen(rd());
        constexpr float bias_std = 0.0f;  // Currently 0, but prepared for future use
        std::uniform_real_distribution<> dis(-bias_std, bias_std);
        q_bias_(2) = dis(gen);
        q_bias_(3) = dis(gen);
        q_bias_(4) = dis(gen);
        q_bias_(8) = dis(gen);
        q_bias_(9) = dis(gen);
        q_bias_(10) = dis(gen);
    }
}

void RlController::processBias()
{
    // Apply bias to joint positions (vectorized)
    q_noise_ += q_bias_;
}

void RlController::processObservation() // [linvel, angvel, proj_grav, commands, dof_pos, dof_vel, actions]
{
    static int obs_call_count = 0;
    obs_call_count++;
    const bool should_debug = debug_enabled_ && (obs_call_count % debug_print_frequency_ == 1);

    int data_idx = 0;

    // Construct quaternion from virtual joint state (w, x, y, z order)
    const Eigen::Quaterniond q(
        dc_cc_.rd_.q_virtual_(6),   // w
        dc_cc_.rd_.q_virtual_(3),   // x
        dc_cc_.rd_.q_virtual_(4),   // y
        dc_cc_.rd_.q_virtual_(5)    // z
    );

    const Eigen::Quaterniond q_conj = q.conjugate();
    // Optimization: Use noalias() to avoid temporary allocation
    base_lin_vel_.noalias() = q_conj * dc_cc_.rd_.q_dot_virtual_.segment<3>(0);
    // CRITICAL FIX: Transform angular velocity from world frame to body frame
    // Python training uses: base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])
    // q.conjugate() * v is equivalent to quat_rotate_inverse(q, v)
    base_ang_vel_.noalias() = q_conj * dc_cc_.rd_.q_dot_virtual_.segment<3>(3); 
    // base_ang_vel_.noalias() = dc_cc_.rd_.q_dot_virtual_.segment<3>(3); 

    //? Copy angular velocity to state
    state_cur_[data_idx++] = base_ang_vel_(0);
    state_cur_[data_idx++] = base_ang_vel_(1);
    state_cur_[data_idx++] = base_ang_vel_(2);

    //? Compute projected gravity
    static const Eigen::Vector3d grav(0, 0, -1.);
    const Eigen::Vector3d projected_grav = q_conj * grav;

    state_cur_[data_idx++] = projected_grav(0);
    state_cur_[data_idx++] = projected_grav(1);
    state_cur_[data_idx++] = projected_grav(2);

    //? Compute heading and update commands
    static const Eigen::Vector3d forward_vec(1., 0, 0);
    const Eigen::Vector3d forward = q * forward_vec;
    const double heading = atan2(forward(1), forward(0));

    state_cur_[data_idx++] = commands_(0);
    state_cur_[data_idx++] = commands_(1);
    // Optimization: Use std::clamp for heading control (more readable)
    if (heading_mode_) {
        const double heading_error = target_heading_ - heading;
        commands_(2) = std::clamp(2.0 * heading_error, -1.0, 1.0);
    }
    state_cur_[data_idx++] = commands_(2);

    //? Joint positions relative to initial (optimized: reduced loop overhead)
    // Unroll critical loop for better performance
    for (int i = 0; i < num_actuator_action; ++i) {
        state_cur_[data_idx++] = q_noise_(i) - q_init_(i);
    }

    ///? Joint velocities
    for (int i = 0; i < num_actuator_action; ++i) {
        state_cur_[data_idx++] = q_vel_noise_(i);
    }

    // Compute step period based on velocity magnitude (matching Python training)
    // Python: step_period = clip(0.3 / sqrt(vx^2 + vy^2), min=0.3, max=0.6)
    const double prev_step_period = step_period_;
    constexpr double eps = 1.e-6;
    const double vel_magnitude = std::sqrt(commands_(0)*commands_(0) + commands_(1)*commands_(1));
    step_period_ = std::clamp(0.3 / (vel_magnitude + eps), 0.3, 0.6);
    step_ticks_ *= step_period_ / prev_step_period;

    //? Phase information
    // phase_indicator_ = 0 means left stance, right swing (right foot moves first)
    // phase_indicator_ = 1 means right stance, left swing (left foot moves first)
    const double phase_angle = 2.0 * M_PI * (step_ticks_ + phase_indicator_ * step_period_) / (2.0 * step_period_);
    state_cur_[data_idx++] = cos(phase_angle);
    state_cur_[data_idx++] = sin(phase_angle);

    // Debug: Print phase info periodically
    if (should_debug) {
        DEBUG_INFO("Phase: indicator=" + std::to_string(phase_indicator_) + 
                   " (0=R_swing, 1=L_swing), ticks=" + std::to_string(step_ticks_) + 
                   ", period=" + std::to_string(step_period_) + 
                   ", angle=" + std::to_string(phase_angle) +
                   ", cos=" + std::to_string(cos(phase_angle)) + 
                   ", sin=" + std::to_string(sin(phase_angle)));
    }

    //? Previous actions (use std::clamp for consistency and readability)
    for (int i = 0; i < num_actuator_action; ++i) {
        state_cur_[data_idx++] = static_cast<float>(std::clamp(rl_action_(i), -1.0, 1.0));
    }
    
    assert(data_idx == num_cur_state);

    // ===== CLIP OBSERVATIONS: Match Python training clip_observations = 100.0 =====
    // Optimization: Use std::clamp for better compiler optimization and readability
    constexpr float clip_observations = 100.0f;
    constexpr float clip_min = -clip_observations;
    constexpr float clip_max = clip_observations;

    // Process in chunks for potential SIMD optimization
    int i = 0;
    for (; i + 3 < num_cur_state; i += 4) {
        state_cur_[i]   = std::clamp(state_cur_[i],   clip_min, clip_max);
        state_cur_[i+1] = std::clamp(state_cur_[i+1], clip_min, clip_max);
        state_cur_[i+2] = std::clamp(state_cur_[i+2], clip_min, clip_max);
        state_cur_[i+3] = std::clamp(state_cur_[i+3], clip_min, clip_max);
    }
    // Handle remaining elements
    for (; i < num_cur_state; ++i) {
        state_cur_[i] = std::clamp(state_cur_[i], clip_min, clip_max);
    }

    // ===== VALIDATION: Check for NaN/Inf in observations =====
    validateAndFixNaN(state_cur_, num_cur_state, 0.0f);

    // Debug: Print observation details
    if (should_debug) {
        DEBUG_HEADER("PROCESSING OBSERVATION #" + std::to_string(obs_call_count));
        
        // Quaternion info
        std::cout << COLOR_MAGENTA << "[QUAT] " << COLOR_WHITE
                << "q(w,x,y,z) = (" << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << ")"
                << " |q| = " << q.norm()
                << COLOR_RESET << std::endl;
        
        // Raw observations
        std::cout << COLOR_BLUE << "[OBS RAW] Angular vel (body frame): [" << std::fixed << std::setprecision(3) << base_ang_vel_(0) << ", " << base_ang_vel_(1) << ", " << base_ang_vel_(2) << "]" << COLOR_RESET << std::endl;
        std::cout << COLOR_BLUE << "[OBS RAW] Projected gravity: [" << projected_grav(0) << ", " << projected_grav(1) << ", " << projected_grav(2) << "]" << COLOR_RESET << std::endl;
        std::cout << COLOR_BLUE << "[OBS RAW] Commands: [vel_x=" << commands_(0) << ", vel_y=" << commands_(1) << ", vel_yaw=" << commands_(2) << "]" << COLOR_RESET << std::endl;
        std::cout << COLOR_BLUE << "[OBS RAW] Joint pos delta (first 6): [";
        for (int i = 0; i < 6; ++i) {
            std::cout << (q_noise_(i) - q_init_(i));
            if (i < 5) std::cout << ", ";
        }
        std::cout << "]" << COLOR_RESET << std::endl;
        std::cout << COLOR_BLUE << "[OBS RAW] Joint vel (first 6): [";
        for (int i = 0; i < 6; ++i) {
            std::cout << q_vel_noise_(i);
            if (i < 5) std::cout << ", ";
        }
        std::cout << "]" << COLOR_RESET << std::endl;

        // Left vs right leg comparison
        std::cout << COLOR_CYAN << "[DEBUG] Left leg joint positions (actual): [";
        for (int i = 0; i < 6; ++i) {
            std::cout << std::fixed << std::setprecision(3) << q_noise_(i);
            if (i < 5) std::cout << ", ";
        }
        std::cout << "]" << COLOR_RESET << std::endl;
        std::cout << COLOR_CYAN << "[DEBUG] Right leg joint positions (actual): [";
        for (int i = 6; i < 12; ++i) {
            std::cout << std::fixed << std::setprecision(3) << q_noise_(i);
            if (i < 11) std::cout << ", ";
        }
        std::cout << "]" << COLOR_RESET << std::endl;
        
        // Observation vector and stats
        DEBUG_SUCCESS("Observation vector constructed: " + std::to_string(num_cur_state) + " elements");
        std::cout << COLOR_MAGENTA << "[OBS] First 47 elements: " << COLOR_WHITE;
        for (int i = 0; i < std::min(47, num_cur_state); i++) {
            std::cout << state_cur_[i] << " ";
        }
        std::cout << COLOR_RESET << std::endl;

        float obs_min = *std::min_element(state_cur_.begin(), state_cur_.end());
        float obs_max = *std::max_element(state_cur_.begin(), state_cur_.end());
        float obs_sum = std::accumulate(state_cur_.begin(), state_cur_.end(), 0.0f);
        float obs_mean = obs_sum / num_cur_state;
        std::cout << COLOR_CYAN << "[OBS_STATS] Min: " << obs_min
                << ", Max: " << obs_max
                << ", Mean: " << obs_mean
                << COLOR_RESET << std::endl;
    }

    // Copy state to critic state (with padding zeros if needed) - SAFE memcpy (both float)
    // Optimization: Use memcpy for faster bulk copy of POD types (both vectors are float)
    const size_t copy_size = std::min(num_cur_state, num_cur_critic_state);
    std::memcpy(critic_state_cur_.data(), state_cur_.data(), copy_size * sizeof(float));
    if (num_cur_critic_state > num_cur_state) {
        std::memset(critic_state_cur_.data() + num_cur_state, 0, (num_cur_critic_state - num_cur_state) * sizeof(float));
    }

    // Normalize the state if normalizer exists
    auto* normalizer_session = getSession("normalizer");
    if (normalizer_session && !input_states_buffer_n.empty()) {
        // Verify buffer sizes before copy
        if (num_cur_critic_state > input_states_buffer_n[0].size()) {
            RCLCPP_ERROR(dc_.node_->get_logger(), "Buffer overflow: trying to copy %zu elements into buffer of size %zu",
                        num_cur_critic_state, input_states_buffer_n[0].size());
            return;
        }

        // Optimization: Use memcpy instead of std::copy (both float vectors)
        std::memcpy(input_states_buffer_n[0].data(), critic_state_cur_.data(), num_cur_critic_state * sizeof(float));
        output_tensors_n = normalizer_session->Run(Ort::RunOptions{nullptr}, 
                                                input_names_char_n.data(), 
                                                input_tensors_n.data(), 
                                                input_number_n, 
                                                output_names_char_n.data(), 
                                                output_number_n);

        // Verify output tensor exists and has data
        if (output_tensors_n.empty() || !output_tensors_n[0].IsTensor()) {
            RCLCPP_ERROR(dc_.node_->get_logger(), "Normalizer output tensor is invalid");
            return;
        }

        auto* output_data = output_tensors_n[0].GetTensorMutableData<float>();
        if (!output_data) {
            RCLCPP_ERROR(dc_.node_->get_logger(), "Normalizer output data pointer is null");
            return;
        }

        // Optimization: Use memcpy for bulk copy of normalized state (both float)
        std::memcpy(normalized_state_cur_.data(), output_data, num_cur_critic_state * sizeof(float));

        // Optimization: Use memcpy instead of std::copy (both float vectors)
        std::memcpy(input_states_buffer[input_obs_idx_].data(), normalized_state_cur_.data(), num_cur_state * sizeof(float));

    } else {
        // If no normalizer, use raw state - optimized with memcpy (both float)
        std::memcpy(input_states_buffer[input_obs_idx_].data(), state_cur_.data(), num_cur_state * sizeof(float));
    }

    // Optimization: Use memcpy for hidden state transfer (both float)
    std::memcpy(input_states_buffer[input_h0_idx_].data(), h_cur_.data(), num_cur_h * sizeof(float));

    if (should_debug) {
        // Normalizer debug
        if (normalizer_session && !input_states_buffer_n.empty()) {
            DEBUG_INFO("Running normalizer...");
            DEBUG_DATA("num_cur_critic_state", num_cur_critic_state);
            DEBUG_DATA("input_states_buffer_n[0].size()", input_states_buffer_n[0].size());
            std::cout << COLOR_MAGENTA << "[NORM_OBS] First 47 elements: " << COLOR_WHITE;
            for (int i = 0; i < std::min(47, (int)num_cur_state); i++) {
                std::cout << normalized_state_cur_[i] << " ";
            }
            std::cout << COLOR_RESET << std::endl;
        } else {
            DEBUG_INFO("Using raw observation (no normalizer)");
        }
        
        // Hidden state debug
        std::cout << COLOR_MAGENTA << "[H_STATE] First 10 h_cur_ elements: " << COLOR_WHITE;
        for (int i = 0; i < std::min(10, (int)num_cur_h); i++) {
            std::cout << h_cur_[i] << " ";
        }
        std::cout << COLOR_RESET << std::endl;
    }
}

void RlController::feedforwardPolicy()
{
    static int policy_call_count = 0;
    policy_call_count++;
    const bool should_debug = debug_enabled_ && (policy_call_count % debug_print_frequency_ == 1);

    auto* actor_session = getSession("actor");
    if (!actor_session) {
        DEBUG_ERROR("Actor session not found!");
        RCLCPP_ERROR(dc_.node_->get_logger(), "Actor session not found!");
        if (rl_action_.rows() != num_action || rl_action_.cols() != 1) {
            rl_action_.resize(num_action, 1);
        }
        rl_action_.setZero();
        return;
    }

    // Optimization: Reuse RunOptions to avoid repeated allocation
    static Ort::RunOptions run_options{nullptr};
    output_tensors = actor_session->Run(run_options,
                                        input_names_char.data(),
                                        input_tensors.data(),
                                        input_number,
                                        output_names_char.data(),
                                        output_number);

    float* action_data = output_tensors[output_action_idx_].GetTensorMutableData<float>();

    // Validate and fix NaN/Inf in actions
    validateAndFixNaN(action_data, num_actuator_action, 0.0f);

    // SMOOTH STARTUP: Apply low-pass filter to actions during initial period
    // This prevents sudden jerks from network predictions that haven't stabilized yet
    static Eigen::Matrix<double, Eigen::Dynamic, 1> rl_action_prev_;
    if (rl_action_prev_.rows() != num_actuator_action) {
        rl_action_prev_.resize(num_actuator_action);
        rl_action_prev_.setZero();
    }

    // Low-pass filter: action_filtered = alpha * action_new + (1-alpha) * action_prev
    // Use stronger filtering (lower alpha) during startup, then gradually increase
    constexpr double startup_duration_us = 2.0e6;  // 2 seconds
    double elapsed_us = dc_cc_.rd_.control_time_ * 1.0e6 - start_time_us;
    double alpha = 1.0;  // Default: no filtering after startup

    if (elapsed_us < startup_duration_us) {
        // Ramp alpha from 0.3 (heavy filtering) to 1.0 (no filtering)
        alpha = 0.3 + 0.7 * (elapsed_us / startup_duration_us);
    }

    // Copy validated actions to rl_action_ with optional filtering
    for (size_t i = 0; i < num_actuator_action; ++i) {
        rl_action_(i) = alpha * action_data[i] + (1.0 - alpha) * rl_action_prev_(i);
        rl_action_prev_(i) = rl_action_(i);  // Store for next iteration
    }
    
    if (should_debug) {
        DEBUG_HEADER("FEEDFORWARD POLICY #" + std::to_string(policy_call_count));
        DEBUG_INFO("Running actor network inference...");

        // Print startup ramp info
        if (elapsed_us < startup_duration_us) {
            std::cout << COLOR_CYAN << "[STARTUP RAMP] Progress: " << std::fixed << std::setprecision(1)
                      << (elapsed_us / startup_duration_us * 100.0) << "% | Alpha (filter): "
                      << std::setprecision(3) << alpha << COLOR_RESET << std::endl;
        }

        // Print actions
        std::cout << COLOR_MAGENTA << "[ACTION] All 12 leg actions (filtered): " << COLOR_WHITE;
        for (int i = 0; i < std::min(12, num_actuator_action); ++i) {
            std::cout << std::fixed << std::setprecision(4) << rl_action_(i) << " ";
        }
        std::cout << COLOR_RESET << std::endl;

        // Print action statistics
        double action_mean = 0.0, action_max = -1e9, action_min = 1e9;
        for (int i = 0; i < std::min(12, num_actuator_action); ++i) {
            action_mean += std::abs(rl_action_(i));
            action_max = std::max(action_max, std::abs(rl_action_(i)));
            action_min = std::min(action_min, std::abs(rl_action_(i)));
        }
        action_mean /= std::min(12, num_actuator_action);
        std::cout << COLOR_YELLOW << "[ACTION STATS] Mean(abs): " << action_mean
                  << " | Max(abs): " << action_max << " | Min(abs): " << action_min << COLOR_RESET << std::endl;

        // Print hidden state output for debugging
        float* hn_data = output_tensors[output_hn_idx_].GetTensorMutableData<float>();
        std::cout << COLOR_CYAN << "[HN_OUTPUT] First 10 elements: " << COLOR_WHITE;
        for (int i = 0; i < std::min(10, (int)num_cur_h); ++i) {
            std::cout << hn_data[i] << " ";
        }
        std::cout << COLOR_RESET << std::endl;
    }
}

void RlController::processOthers()
{
    // Update hidden state from network output with validation
    float* hn_data = output_tensors[output_hn_idx_].GetTensorMutableData<float>();

    // Validate and fix NaN/Inf in hidden state
    validateAndFixNaN(hn_data, num_cur_h, 0.0f);

    // Optimization: Copy and clamp hidden state with SIMD-friendly loop
    // Process 4 elements at a time for better vectorization
    float h_max_abs = 0.0f;
    constexpr float h_clamp_min = -10.0f;
    constexpr float h_clamp_max = 10.0f;

    size_t i = 0;
    // Process in chunks of 4 for potential SIMD optimization (use std::clamp for better codegen)
    for (; i + 3 < num_cur_h; i += 4) {
        h_cur_[i]   = std::clamp(hn_data[i],   h_clamp_min, h_clamp_max);
        h_cur_[i+1] = std::clamp(hn_data[i+1], h_clamp_min, h_clamp_max);
        h_cur_[i+2] = std::clamp(hn_data[i+2], h_clamp_min, h_clamp_max);
        h_cur_[i+3] = std::clamp(hn_data[i+3], h_clamp_min, h_clamp_max);

        h_max_abs = std::max({h_max_abs, std::abs(h_cur_[i]), std::abs(h_cur_[i+1]),
                              std::abs(h_cur_[i+2]), std::abs(h_cur_[i+3])});
    }
    // Handle remaining elements
    for (; i < num_cur_h; ++i) {
        h_cur_[i] = std::clamp(hn_data[i], h_clamp_min, h_clamp_max);
        h_max_abs = std::max(h_max_abs, std::abs(h_cur_[i]));
    }

    // Warn if hidden states are getting large (potential divergence)
    if (h_max_abs > 5.0f) {
        RCLCPP_WARN_THROTTLE(dc_.node_->get_logger(), *dc_.node_->get_clock(), 2000,
            "Hidden state magnitude is large: %.3f (may indicate divergence)", h_max_abs);
    }

    // Update walking tick counter
    walking_tick++;

    // Update time tracking for next inference (in microseconds)
    time_inference_pre_us = dc_cc_.rd_.control_time_ * 1e6;

    // Data logging
    if (is_write_file_ && walking_tick % save_frequency_ == 0) {
        writeData();
    }
}

void RlController::updateNextStep()
{
    step_ticks_ += del_t;
    if (step_ticks_ >= step_period_) {
        step_ticks_ = 0.0;
        phase_indicator_ = 1 - phase_indicator_;
    }
}

void RlController::joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    // Debug: First callback notification
    static bool first_callback = true;
    if (first_callback) {
        RCLCPP_INFO(dc_.node_->get_logger(),
            COLOR_BOLD COLOR_GREEN "[JOY] First joystick message received! Callback is working." COLOR_RESET);
        first_callback = false;
    }

    // Check if we have enough axes and buttons
    if (msg->axes.size() < 2 || msg->buttons.size() < 8) {
        RCLCPP_WARN_THROTTLE(dc_.node_->get_logger(), *dc_.node_->get_clock(), 1000,
            "Joy message does not have enough axes or buttons (axes: %zu, buttons: %zu)",
            msg->axes.size(), msg->buttons.size());
        return;
    }

    // Update velocity commands from joystick axes
    commands_(0) = DyrosMath::minmax_cut(vel_scale_x_ * msg->axes[1], -0.5, 1.0);
    commands_(1) = DyrosMath::minmax_cut(vel_scale_y_ * msg->axes[0], -0.8, 0.8);

    // Button 1: Increase velocity scaling
    if (msg->buttons[1] == 1 && vel_scale_x_ < 1.0 && vel_scale_y_ < 0.3) {
        vel_scale_x_ += 0.03;
        vel_scale_y_ += 0.01;
        RCLCPP_INFO(dc_.node_->get_logger(), "Velocity X: %.2f, Velocity Y: %.2f", vel_scale_x_, vel_scale_y_);
    }

    // Button 0: Decrease velocity scaling
    if (msg->buttons[0] == 1 && vel_scale_x_ > 0.1 && vel_scale_y_ > 0.03) {
        vel_scale_x_ -= 0.03;
        vel_scale_y_ -= 0.01;
        RCLCPP_INFO(dc_.node_->get_logger(), "Velocity X: %.2f, Velocity Y: %.2f", vel_scale_x_, vel_scale_y_);
    }

    // Button 6: Turn right
    if (msg->buttons[6] == 1) {
        commands_(2) = 0.6;
    }
    // Button 7: Turn left
    else if (msg->buttons[7] == 1) {
        commands_(2) = -0.6;
    }
    // No turn buttons pressed
    else {
        commands_(2) = 0.0;
    }

    // Debug output: Print joystick commands (throttled to avoid spam)
    static int joy_debug_counter = 0;
    if (debug_enabled_ && ++joy_debug_counter % 30 == 0) {  // Print every 30 messages (~1 Hz at 30Hz joy rate)
        std::cout << COLOR_BOLD COLOR_CYAN
            << "[JOY CALLBACK] Commands updated: vel_x=" << commands_(0)
            << ", vel_y=" << commands_(1)
            << ", vel_yaw=" << commands_(2)
            << " | Raw axes: [" << msg->axes[1] << ", " << msg->axes[0]
            << "] | Scales: [" << vel_scale_x_ << ", " << vel_scale_y_ << "]"
            << COLOR_RESET << std::endl;
    }
}

void RlController::openDataFile()
{
    if (!data_output_enabled_) {
        return;
    }
    
    // Create directory if it doesn't exist
    std::filesystem::create_directories(data_output_dir_);
    
    // Generate filename with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << data_output_dir_ << "/" << data_filename_prefix_ << "_" 
       << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S") << ".csv";
    
    std::string filename = ss.str();
    writeFile.open(filename);
    
    if (!writeFile.is_open()) {
        is_write_file_ = false;
        RCLCPP_ERROR(dc_.node_->get_logger(), "Failed to open data file: %s", filename.c_str());
        return;
    }

    is_write_file_ = true;
    RCLCPP_INFO(dc_.node_->get_logger(), "CSV data file opened: %s", filename.c_str());
    
    // Use helper lambda for repetitive column headers
    auto writeColumns = [this](const std::string& prefix, int count, int offset = 0) {
        for (int i = 0; i < count; i++) {
            writeFile << "," << prefix << (i + offset);
        }
    };
    
    // Write CSV header in one efficient stream
    writeFile << "time_since_inference";
    writeFile << ",time_current";
    
    
    // writeColumns("LF_CF_FT_", 6);
    // writeColumns("RF_CF_FT_", 6);
    writeColumns("torque_desired_", MODEL_DOF);
    writeColumns("q_noise_", MODEL_DOF);
    writeColumns("q_dot_lpf_", MODEL_DOF);
    writeFile << ",base_lin_vel_x,base_lin_vel_y,base_lin_vel_z";
    writeFile << ",base_ang_vel_x,base_ang_vel_y,base_ang_vel_z";
    writeColumns("q_dot_virtual_", MODEL_DOF, 6);  // q_dot_virtual from index 6 to 6+MODEL_DOF-1
    writeColumns("q_virtual_", MODEL_DOF_QVIRTUAL);
    writeFile << ",heading,value,stop_by_value_thres,command_x,command_y,command_yaw\n";
}

void RlController::writeData()
{
    if (!is_write_file_ || !writeFile.is_open()) {
        return;
    }
    
    // Compute heading for logging
    // Eigen::Quaterniond constructor takes (w, x, y, z) order
    const Eigen::Quaterniond q(
        dc_cc_.rd_.q_virtual_(6),                       // w
        dc_cc_.rd_.q_virtual_(3),                       // x
        dc_cc_.rd_.q_virtual_(4),                       // y
        dc_cc_.rd_.q_virtual_(5)                        // z
    );
    static const Eigen::Vector3d forward_vec(1., 0, 0);
    const Eigen::Vector3d forward = q * forward_vec;
    const double heading = atan2(forward(1), forward(0));
    
    // Helper lambda for writing vector/matrix data
    auto writeVec = [this](const auto& vec, int size) {
        for (int i = 0; i < size; i++) {
            writeFile << "," << vec(i);
        }
    };
    
    // Write data in CSV format - one efficient stream operation
    writeFile << std::fixed << std::setprecision(6);
    writeFile << (dc_cc_.rd_.control_time_ * 1e6 - time_inference_pre_us) / 1e6;
    writeFile << "," << dc_cc_.rd_.control_time_;
    
    // writeVec(dc_cc_.rd_.LF_CF_FT, dc_cc_.rd_.LF_CF_FT.size());
    // writeVec(dc_cc_.rd_.RF_CF_FT, dc_cc_.rd_.RF_CF_FT.size());
    writeVec(dc_cc_.rd_.torque_desired, MODEL_DOF);
    writeVec(q_noise_, MODEL_DOF);
    writeVec(q_dot_lpf_, MODEL_DOF);
    writeVec(base_lin_vel_, 3);
    writeVec(base_ang_vel_, 3);
    
    // q_dot_virtual segment from index 6 to 6+MODEL_DOF-1
    for (int i = 6; i < 6 + MODEL_DOF; i++) {
        writeFile << "," << dc_cc_.rd_.q_dot_virtual_(i);
    }
    
    writeVec(dc_cc_.rd_.q_virtual_, MODEL_DOF_QVIRTUAL);
    
    writeFile << "," << heading 
            << "," << value_ 
            << "," << stop_by_value_thres_
            << "," << commands_(0) 
            << "," << commands_(1) 
            << "," << commands_(2) << "\n";
    
}