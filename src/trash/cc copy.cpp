#include "cc.h"
#include <algorithm>
#include <filesystem>
#include <map>
#include <fstream>
#include <iomanip>
#include <yaml-cpp/yaml.h>

using namespace TOCABI;

CustomController::CustomController(RobotData &rd) : rd_(rd)
{
    std::cout << "========== INITIALIZING CUSTOM CONTROLLER ==========" << std::endl;

    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "tocabi");
    memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    std::cout << "[✓] ONNX Runtime environment created" << std::endl;

    // Default config file path
    config_file_path_ = "/home/rui/ubuntu-20-04/raibertGRU_ws/src/tocabi_cc/config/cc_config.yaml";

    std::cout << "[INFO] Loading configuration from: " << config_file_path_ << std::endl;

    // Initialize variables and load configuration (single cohesive function)
    initVariablesAndLoadConfig(config_file_path_);

    // Create joy subscriber
    joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy_wh", 10, &CustomController::joyCallback, this);
    std::cout << "[✓] Joy subscriber created on topic: /joy_wh" << std::endl;

    ControlVal_.setZero();
}

void CustomController::initVariablesAndLoadConfig(const std::string& config_file_path)
{
    std::cout << "========== INITIALIZING CONTROLLER ==========" << std::endl;

    //===== STAGE 1: Initialize config-independent variables =====
    std::cout << "===== Stage 1: Config-independent initialization =====" << std::endl;

    // Initialize action vector
    rl_action_.resize(num_action, 1);
    rl_action_.setZero();
    std::cout << "[✓] Action vector initialized: " << num_action << " dimensions" << std::endl;

    // Initialize state vectors for RL observation
    state_cur_.assign(num_cur_state, 0.0f);
    critic_state_cur_.assign(num_cur_critic_state, 0.0f);
    normalized_state_cur_.assign(num_cur_state, 0.0f);
    normalized_critic_state_cur_.assign(num_cur_critic_state, 0.0f);
    h_cur_.assign(num_cur_h, 0.0f);
    latent_cur_.assign(num_cur_latent, 0.0f);
    std::cout << "[✓] State vectors initialized" << std::endl;

    // Initialize robot state variables
    q_noise_.setZero();
    q_noise_pre_.setZero();
    q_vel_noise_.setZero();
    q_dot_lpf_.setZero();
    torque_init_.setZero();
    torque_spline_.setZero();
    torque_rl_.setZero();
    q_stop_.setZero();
    std::cout << "[✓] Robot state variables initialized" << std::endl;

    // Initialize velocity tracking
    base_lin_vel.setZero();
    base_ang_vel.setZero();
    std::cout << "[✓] Velocity tracking initialized" << std::endl;

    // Initialize walking/stepping variables
    step_period_ = 0.8;
    step_ticks_ = 0.0;
    phase_indicator_ = 0;
    std::cout << "[✓] Walking parameters initialized" << std::endl;

    // Initialize command and control variables
    commands_.setZero();
    target_heading_ = 0.0;
    heading_mode_ = false;
    std::cout << "[✓] Command variables initialized" << std::endl;

    // Initialize joint bias
    q_bias_.setZero();

    // Default values (will be overridden by config file if present)
    q_init_.setZero();
    torque_bound_.setConstant(80.0);

    // PD gains will be loaded from config
    kp_.setIdentity();
    kv_.setIdentity();

    // pd_limit will be loaded from config
    pd_limit.setConstant(-1.0);
    pd_limit.col(1).setConstant(1.0);
    std::cout << "[✓] PD limits initialized to [-1.0, 1.0]" << std::endl;

    //===== STAGE 2: Load configuration and override defaults =====
    std::cout << "===== Stage 2: Loading configuration file =====" << std::endl;

    try {
        const YAML::Node config = YAML::LoadFile(config_file_path);

        // Get source directory for resolving relative paths
        std::string source_dir = "/home/rui/ubuntu-20-04/raibertGRU_ws/src/tocabi_cc";

        // Load model paths
        std::map<std::string, std::string> model_paths;
        if (config["models"]) {
            for (const auto& model : config["models"]) {
                const std::string name = model.first.as<std::string>();
                std::string path = model.second.as<std::string>();

                // Resolve relative paths to absolute paths
                if (!path.empty() && path[0] != '/') {
                    if (path.find('/') == std::string::npos) {
                        path = source_dir + "/onnx/" + path;
                    } else {
                        path = source_dir + "/" + path;
                    }
                }

                model_paths[name] = path;
                std::cout << "[INFO] Model '" << name << "': " << path << std::endl;
            }
        }

        // Load data output configuration
        if (config["data_output"]) {
            const auto& data_out = config["data_output"];
            data_output_enabled_ = data_out["enabled"].as<bool>(true);
            data_output_dir_ = data_out["output_dir"].as<std::string>("");

            if (data_output_dir_.empty()) {
                data_output_dir_ = source_dir + "/result";
            } else if (data_output_dir_[0] != '/') {
                data_output_dir_ = source_dir + "/" + data_output_dir_;
            }

            data_filename_prefix_ = data_out["filename_prefix"].as<std::string>("data");
            save_frequency_ = data_out["save_frequency"].as<int>(100);

            std::cout << "[INFO] Data output enabled: " << (data_output_enabled_ ? "true" : "false") << std::endl;
            std::cout << "[INFO] Data output directory: " << data_output_dir_ << std::endl;

            if (data_output_enabled_) {
                std::filesystem::create_directories(data_output_dir_);

                // Open data file
                std::string full_path = data_output_dir_ + "/" + data_filename_prefix_ + ".csv";
                writeFile.open(full_path, std::ofstream::out);
                writeFile << std::fixed << std::setprecision(8);

                // Write CSV header
                writeFile << "time\t"
                          << "LF_FT_fx\tLF_FT_fy\tLF_FT_fz\tLF_FT_tx\tLF_FT_ty\tLF_FT_tz\t"
                          << "RF_FT_fx\tRF_FT_fy\tRF_FT_fz\tRF_FT_tx\tRF_FT_ty\tRF_FT_tz\t"
                          << "torque_des_0\ttorque_des_1\ttorque_des_2\ttorque_des_3\ttorque_des_4\ttorque_des_5\t"
                          << "torque_des_6\ttorque_des_7\ttorque_des_8\ttorque_des_9\ttorque_des_10\ttorque_des_11\t"
                          << "torque_des_12\ttorque_des_13\ttorque_des_14\ttorque_des_15\ttorque_des_16\ttorque_des_17\t"
                          << "torque_des_18\ttorque_des_19\ttorque_des_20\ttorque_des_21\ttorque_des_22\ttorque_des_23\t"
                          << "torque_des_24\ttorque_des_25\ttorque_des_26\ttorque_des_27\ttorque_des_28\ttorque_des_29\t"
                          << "torque_des_30\ttorque_des_31\ttorque_des_32\t"
                          << "q_0\tq_1\tq_2\tq_3\tq_4\tq_5\tq_6\tq_7\tq_8\tq_9\tq_10\tq_11\t"
                          << "q_12\tq_13\tq_14\tq_15\tq_16\tq_17\tq_18\tq_19\tq_20\tq_21\tq_22\tq_23\t"
                          << "q_24\tq_25\tq_26\tq_27\tq_28\tq_29\tq_30\tq_31\tq_32\t"
                          << "qdot_lpf_0\tqdot_lpf_1\tqdot_lpf_2\tqdot_lpf_3\tqdot_lpf_4\tqdot_lpf_5\t"
                          << "qdot_lpf_6\tqdot_lpf_7\tqdot_lpf_8\tqdot_lpf_9\tqdot_lpf_10\tqdot_lpf_11\t"
                          << "qdot_lpf_12\tqdot_lpf_13\tqdot_lpf_14\tqdot_lpf_15\tqdot_lpf_16\tqdot_lpf_17\t"
                          << "qdot_lpf_18\tqdot_lpf_19\tqdot_lpf_20\tqdot_lpf_21\tqdot_lpf_22\tqdot_lpf_23\t"
                          << "qdot_lpf_24\tqdot_lpf_25\tqdot_lpf_26\tqdot_lpf_27\tqdot_lpf_28\tqdot_lpf_29\t"
                          << "qdot_lpf_30\tqdot_lpf_31\tqdot_lpf_32\t"
                          << "base_lin_vel_x\tbase_lin_vel_y\tbase_lin_vel_z\t"
                          << "base_ang_vel_x\tbase_ang_vel_y\tbase_ang_vel_z\t"
                          << "qdot_0\tqdot_1\tqdot_2\tqdot_3\tqdot_4\tqdot_5\tqdot_6\tqdot_7\tqdot_8\tqdot_9\t"
                          << "qdot_10\tqdot_11\tqdot_12\tqdot_13\tqdot_14\tqdot_15\tqdot_16\tqdot_17\tqdot_18\tqdot_19\t"
                          << "qdot_20\tqdot_21\tqdot_22\tqdot_23\tqdot_24\tqdot_25\tqdot_26\tqdot_27\tqdot_28\tqdot_29\t"
                          << "qdot_30\tqdot_31\tqdot_32\t"
                          << "qvirt_0\tqvirt_1\tqvirt_2\tqvirt_3\tqvirt_4\tqvirt_5\tqvirt_6\tqvirt_7\tqvirt_8\tqvirt_9\t"
                          << "qvirt_10\tqvirt_11\tqvirt_12\tqvirt_13\tqvirt_14\tqvirt_15\tqvirt_16\tqvirt_17\tqvirt_18\tqvirt_19\t"
                          << "qvirt_20\tqvirt_21\tqvirt_22\tqvirt_23\tqvirt_24\tqvirt_25\tqvirt_26\tqvirt_27\tqvirt_28\tqvirt_29\t"
                          << "qvirt_30\tqvirt_31\tqvirt_32\tqvirt_33\tqvirt_34\tqvirt_35\tqvirt_36\tqvirt_37\tqvirt_38\t"
                          << "heading\t"
                          << "value\tstop_flag\t"
                          << "cmd_x\tcmd_y\tcmd_yaw\t"
                          << "rf_x\trf_y\trf_z\t"
                          << "lf_x\tlf_y\tlf_z"
                          << std::endl;

                std::cout << "[✓] Data file opened: " << full_path << std::endl;
            }
        }

        // Load controller settings
        if (config["controller"]) {
            ctrl_type_ = config["controller"]["action_type"].as<std::string>("T");
            is_on_robot_ = config["controller"]["is_on_robot"].as<bool>(false);
            ctrl_mode = config["controller"]["ctrl_mode"].as<int>(0);

            if (config["controller"]["inference_frequency"]) {
                hz_ = config["controller"]["inference_frequency"].as<double>(125.0);
                std::cout << "[INFO] Inference frequency: " << hz_ << " Hz" << std::endl;
            }

            std::cout << "[INFO] Controller action type: " << ctrl_type_ << std::endl;
            std::cout << "[INFO] Is on robot: " << (is_on_robot_ ? "true" : "false") << std::endl;
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
                std::cout << "[INFO] Loaded torque bounds for " << torque_bounds.size() << " joints" << std::endl;
            }

            // Load kp gains
            if (params["kp_diagonal"] && params["kp_scale"]) {
                auto kp_values = params["kp_diagonal"].as<std::vector<double>>();
                double kp_scale = params["kp_scale"].as<double>(9.0);
                kp_.setZero();
                for (size_t i = 0; i < kp_values.size() && i < MODEL_DOF; ++i) {
                    kp_(i, i) = kp_values[i] / kp_scale;
                }
                std::cout << "[INFO] Loaded kp gains (scale: " << kp_scale << ")" << std::endl;
            }

            // Load kv gains
            if (params["kv_diagonal"] && params["kv_scale"]) {
                auto kv_values = params["kv_diagonal"].as<std::vector<double>>();
                double kv_scale = params["kv_scale"].as<double>(3.0);
                kv_.setZero();
                for (size_t i = 0; i < kv_values.size() && i < MODEL_DOF; ++i) {
                    kv_(i, i) = kv_values[i] / kv_scale;
                }
                std::cout << "[INFO] Loaded kv gains (scale: " << kv_scale << ")" << std::endl;
            }

            // Load q_init
            if (params["q_init"]) {
                auto q_init_values = params["q_init"].as<std::vector<double>>();
                for (size_t i = 0; i < q_init_values.size() && i < MODEL_DOF; ++i) {
                    q_init_(i) = q_init_values[i];
                }
                std::cout << "[INFO] Loaded initial joint positions (" << q_init_values.size() << " values)" << std::endl;
            }

            // Load pd_limits and pre-compute values for optimization
            if (params["pd_limit"]) {
                auto pd_limits = params["pd_limit"].as<std::vector<std::vector<double>>>();
                for (size_t i = 0; i < pd_limits.size() && i < num_action; ++i) {
                    if (pd_limits[i].size() >= 2) {
                        pd_limit(i, 0) = pd_limits[i][0];
                        pd_limit(i, 1) = pd_limits[i][1];

                        // Pre-compute std and bias for fast PD control
                        pd_limit_std_(i) = (pd_limit(i, 1) - pd_limit(i, 0)) * 0.5;
                        pd_limit_bias_(i) = (pd_limit(i, 1) + pd_limit(i, 0)) * 0.5;
                    }
                }
                std::cout << "[INFO] Loaded PD limits for " << pd_limits.size() << " joints (pre-computed std/bias)" << std::endl;
            }
        }

        // Load command configuration
        if (config["commands"]) {
            const auto& commands = config["commands"];

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
                heading_mode_ = true;
            }

            std::cout << "[INFO] Command velocities: x=" << commands_(0) << ", y=" << commands_(1) << ", yaw=" << commands_(2) << std::endl;
        }

        // Load joystick configuration
        if (config["joystick"]) {
            const auto& joy = config["joystick"];
            vel_scale_x_ = joy["vel_scale_x"].as<float>(0.6);
            vel_scale_y_ = joy["vel_scale_y"].as<float>(0.2);
            std::cout << "[INFO] Joystick velocity scales: x=" << vel_scale_x_ << ", y=" << vel_scale_y_ << std::endl;
        }

        // Load walking parameters
        if (config["walking"]) {
            const auto& walking = config["walking"];
            step_period_ = walking["step_period"].as<float>(0.8);
            max_stride_x = walking["max_stride_x"].as<float>(0.4);
            max_stride_y = walking["max_stride_y"].as<float>(0.12);
            max_stride_yaw = walking["max_stride_yaw"].as<float>(0.4);
            std::cout << "[INFO] Walking parameters loaded" << std::endl;
        }

        // Load debug configuration
        if (config["debug"]) {
            const auto& debug = config["debug"];
            debug_enabled_ = debug["enable_debug_prints"].as<bool>(true);
            debug_enabled_torque = debug["enable_torque_debug_prints"].as<bool>(false);
            debug_print_frequency_ = debug["print_frequency"].as<int>(100);

            std::cout << "[INFO] Debug output enabled: " << (debug_enabled_ ? "true" : "false") << std::endl;
        }

        // Load ONNX models
        if (!model_paths.empty()) {
            loadOnnxModel(model_paths);
        } else {
            std::cout << "[WARN] No models specified in configuration file!" << std::endl;
        }

    } catch (const YAML::Exception& e) {
        std::cerr << "[ERROR] Failed to load config file: " << e.what() << std::endl;
        throw;
    }

    //===== STAGE 3: Finalize config-dependent initialization =====
    std::cout << "===== Stage 3: Config-dependent initialization =====" << std::endl;

    // Initialize bias (depends on is_on_robot_ from config)
    initBias();

    // Calculate time step based on loaded inference frequency
    del_t = 1.0 / hz_;
    std::cout << "[✓] Time step calculated from config" << std::endl;
    std::cout << "[INFO] Inference frequency: " << hz_ << " Hz → time step: " << del_t << " s" << std::endl;

    std::cout << "[✓] Controller initialization complete" << std::endl;
}

Ort::Session* CustomController::getSession(const std::string& name)
{
    auto it = sessions_.find(name);
    if (it != sessions_.end()) {
        return it->second.get();
    }
    return nullptr;
}

void CustomController::loadOnnxModel(const std::map<std::string, std::string>& model_paths)
{
    std::cout << "========== LOADING ONNX MODELS ==========" << std::endl;

    Ort::SessionOptions session_options;
    // Read optimization level from config (default: ORT_DISABLE_ALL for deterministic behavior)
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.AddConfigEntry("session.use_deterministic_compute", "1");
    std::cout << "[✓] ONNX session options configured" << std::endl;

    // Load command file if in file-based command mode
    if (ctrl_mode == 1) {
        string cur_path = "/home/rui/ubuntu-20-04/raibertGRU_ws/src/tocabi_cc/";
        if (is_on_robot_) {
            cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/";
        }
        loadCommand(cur_path + "commands.txt");
    }

    // Load all models with error handling
    for (const auto& [name, path] : model_paths) {
        try {
            std::cout << "[INFO] Loading model '" << name << "' from: " << path << std::endl;
            sessions_[name] = std::make_unique<Ort::Session>(*env, path.c_str(), session_options);
            std::cout << "[✓] Model '" << name << "' loaded successfully" << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "[ERROR] Failed to load " << name << " model: " << e.what() << std::endl;
            throw;
        }
    }

    // Get actor session (required)
    Ort::Session* actor_session = getSession("actor");
    if (!actor_session) {
        throw std::runtime_error("Actor model is required but not loaded!");
    }

    Ort::AllocatorWithDefaultOptions allocator;

    // Initialize actor network metadata
    input_number = actor_session->GetInputCount();
    output_number = actor_session->GetOutputCount();
    input_names.resize(input_number);
    output_names.resize(output_number);
    input_names_char.resize(input_number);
    output_names_char.resize(output_number);

    for (size_t i = 0; i < input_number; i++) {
        Ort::AllocatedStringPtr input_name = actor_session->GetInputNameAllocated(i, allocator);
        input_names[i] = input_name.get();
        input_names_char[i] = input_names[i].c_str();
        if (input_names[i] == "obs") {input_obs_idx_ = i;}
        else if (input_names[i] == "h0"){input_h0_idx_ = i;}
    }

    for (size_t i = 0; i < output_number; i++) {
        Ort::AllocatedStringPtr output_name = actor_session->GetOutputNameAllocated(i, allocator);
        output_names[i] = output_name.get();
        output_names_char[i] = output_names[i].c_str();
        if (output_names[i] == "action") {output_action_idx_ = i;}
        else if (output_names[i] == "latent") {output_latent_idx_ = i;}
        else if (output_names[i] == "hn"){output_hn_idx_ = i;}
    }

    std::cout << "[INFO] Actor input names: ";
    std::copy(input_names.begin(), input_names.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;
    std::cout << "[INFO] Actor output names: ";
    std::copy(output_names.begin(), output_names.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;

    // Initialize actor input tensors
    for (size_t i = 0; i < input_number; ++i) {
        Ort::TypeInfo type_info = actor_session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = tensor_info.GetShape();
        std::cout << "[INFO] Actor input " << i << " shape: " << input_shape.size() << std::endl;
        std::vector<float> input_tensor_values(tensor_info.GetElementCount(), 0.0);
        input_states_buffer.push_back(std::move(input_tensor_values));

        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            *memory_info,
            input_states_buffer.back().data(),
            input_states_buffer.back().size(),
            input_shape.data(),
            input_shape.size()));
    }

    // Initialize normalizer network if present
    Ort::Session* normalizer_session = getSession("normalizer");
    if (normalizer_session) {
        input_number_n = normalizer_session->GetInputCount();
        output_number_n = normalizer_session->GetOutputCount();
        input_names_n.resize(input_number_n);
        output_names_n.resize(output_number_n);
        input_names_char_n.resize(input_number_n);
        output_names_char_n.resize(output_number_n);

        for (size_t i = 0; i < input_number_n; i++) {
            Ort::AllocatedStringPtr input_name = normalizer_session->GetInputNameAllocated(i, allocator);
            input_names_n[i] = input_name.get();
            input_names_char_n[i] = input_names_n[i].c_str();
        }

        for (size_t i = 0; i < output_number_n; i++) {
            Ort::AllocatedStringPtr output_name = normalizer_session->GetOutputNameAllocated(i, allocator);
            output_names_n[i] = output_name.get();
            output_names_char_n[i] = output_names_n[i].c_str();
        }

        for (size_t i = 0; i < input_number_n; ++i) {
            Ort::TypeInfo type_info = normalizer_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_shape = tensor_info.GetShape();
            std::vector<float> input_tensor_values(tensor_info.GetElementCount(), 0.0);
            input_states_buffer_n.push_back(std::move(input_tensor_values));

            input_tensors_n.emplace_back(Ort::Value::CreateTensor<float>(
                *memory_info,
                input_states_buffer_n.back().data(),
                input_states_buffer_n.back().size(),
                input_shape.data(),
                input_shape.size()));
        }
        std::cout << "[✓] Normalizer network initialized" << std::endl;
    }

    // Initialize denormalizer network if present
    Ort::Session* denormalizer_session = getSession("denormalizer");
    if (denormalizer_session) {
        input_number_dn = denormalizer_session->GetInputCount();
        output_number_dn = denormalizer_session->GetOutputCount();
        input_names_dn.resize(input_number_dn);
        output_names_dn.resize(output_number_dn);
        input_names_char_dn.resize(input_number_dn);
        output_names_char_dn.resize(output_number_dn);

        for (size_t i = 0; i < input_number_dn; i++) {
            Ort::AllocatedStringPtr input_name = denormalizer_session->GetInputNameAllocated(i, allocator);
            input_names_dn[i] = input_name.get();
            input_names_char_dn[i] = input_names_dn[i].c_str();
        }

        for (size_t i = 0; i < output_number_dn; i++) {
            Ort::AllocatedStringPtr output_name = denormalizer_session->GetOutputNameAllocated(i, allocator);
            output_names_dn[i] = output_name.get();
            output_names_char_dn[i] = output_names_dn[i].c_str();
        }

        for (size_t i = 0; i < input_number_dn; ++i) {
            Ort::TypeInfo type_info = denormalizer_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_shape = tensor_info.GetShape();
            std::vector<float> input_tensor_values(tensor_info.GetElementCount(), 0.0);
            input_states_buffer_dn.push_back(std::move(input_tensor_values));

            input_tensors_dn.emplace_back(Ort::Value::CreateTensor<float>(
                *memory_info,
                input_states_buffer_dn.back().data(),
                input_states_buffer_dn.back().size(),
                input_shape.data(),
                input_shape.size()));
        }
        std::cout << "[✓] Denormalizer network initialized" << std::endl;
    }

    // Initialize decoder network if present
    Ort::Session* decoder_session = getSession("decoder");
    if (decoder_session) {
        input_number_d = decoder_session->GetInputCount();
        output_number_d = decoder_session->GetOutputCount();
        input_names_d.resize(input_number_d);
        output_names_d.resize(output_number_d);
        input_names_char_d.resize(input_number_d);
        output_names_char_d.resize(output_number_d);

        for (size_t i = 0; i < input_number_d; i++) {
            Ort::AllocatedStringPtr input_name = decoder_session->GetInputNameAllocated(i, allocator);
            input_names_d[i] = input_name.get();
            input_names_char_d[i] = input_names_d[i].c_str();
        }

        for (size_t i = 0; i < output_number_d; i++) {
            Ort::AllocatedStringPtr output_name = decoder_session->GetOutputNameAllocated(i, allocator);
            output_names_d[i] = output_name.get();
            output_names_char_d[i] = output_names_d[i].c_str();
        }

        for (size_t i = 0; i < input_number_d; ++i) {
            Ort::TypeInfo type_info = decoder_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_shape = tensor_info.GetShape();
            std::vector<float> input_tensor_values(tensor_info.GetElementCount(), 0.0);
            input_states_buffer_d.push_back(std::move(input_tensor_values));

            input_tensors_d.emplace_back(Ort::Value::CreateTensor<float>(
                *memory_info,
                input_states_buffer_d.back().data(),
                input_states_buffer_d.back().size(),
                input_shape.data(),
                input_shape.size()));
        }
        std::cout << "[✓] Decoder network initialized" << std::endl;
    }

    // Initialize critic network if present
    Ort::Session* critic_session = getSession("critic");
    if (critic_session) {
        input_number_c = critic_session->GetInputCount();
        output_number_c = critic_session->GetOutputCount();
        input_names_c.resize(input_number_c);
        output_names_c.resize(output_number_c);
        input_names_char_c.resize(input_number_c);
        output_names_char_c.resize(output_number_c);

        for (size_t i = 0; i < input_number_c; i++) {
            Ort::AllocatedStringPtr input_name = critic_session->GetInputNameAllocated(i, allocator);
            input_names_c[i] = input_name.get();
            input_names_char_c[i] = input_names_c[i].c_str();
        }

        for (size_t i = 0; i < output_number_c; i++) {
            Ort::AllocatedStringPtr output_name = critic_session->GetOutputNameAllocated(i, allocator);
            output_names_c[i] = output_name.get();
            output_names_char_c[i] = output_names_c[i].c_str();
        }

        for (size_t i = 0; i < input_number_c; ++i) {
            Ort::TypeInfo type_info = critic_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_shape = tensor_info.GetShape();
            std::vector<float> input_tensor_values(tensor_info.GetElementCount(), 0.0);
            input_states_buffer_c.push_back(std::move(input_tensor_values));

            input_tensors_c.emplace_back(Ort::Value::CreateTensor<float>(
                *memory_info,
                input_states_buffer_c.back().data(),
                input_states_buffer_c.back().size(),
                input_shape.data(),
                input_shape.size()));
        }
        std::cout << "[✓] Critic network initialized" << std::endl;
    }

    std::cout << "[✓] All ONNX models loaded and initialized" << std::endl;
}

void CustomController::initBias()
{
    q_bias_.setZero();
    if (~is_on_robot_){
        std::random_device rd;  
        std::mt19937 gen(rd());
        float bias_std = 0.;
        std::uniform_real_distribution<> dis(-bias_std, bias_std);
        q_bias_(2) = dis(gen);
        q_bias_(3) = dis(gen);
        q_bias_(4) = dis(gen);
        q_bias_(8) = dis(gen);
        q_bias_(9) = dis(gen);
        q_bias_(10) = dis(gen);
        // for (int i = 0; i < num_actuator_action; i++){
        //     q_bias_(i) = dis(gen);

        // }
    }
}

void CustomController::processBias()
{
    for (int i = 0; i < MODEL_DOF; i++){
        q_noise_(i) += q_bias_(i);
    }
}

void CustomController::processNoise()
{
    time_cur_ = rd_cc_.control_time_us_ / 1e6;
    if (is_on_robot_)
    {
        q_vel_noise_ = rd_cc_.q_dot_virtual_.segment(6,MODEL_DOF);
        q_noise_= rd_cc_.q_virtual_.segment(6,MODEL_DOF);
        if (time_cur_ - time_pre_ > 0.0)
        {
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 4.0);
        }
        else
        {
            q_dot_lpf_ = q_dot_lpf_;
        }
    }
    else
    {
        std::random_device rd;  
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.00001, 0.00001);
        for (int i = 0; i < MODEL_DOF; i++) {
            q_noise_(i) = rd_cc_.q_virtual_(6+i) + dis(gen);
        }
        if (time_cur_ - time_pre_ > 0.0)
        {
            q_vel_noise_ = (q_noise_ - q_noise_pre_) / (time_cur_ - time_pre_);
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 4.0);
        }
        else
        {
            q_vel_noise_ = q_vel_noise_;
            q_dot_lpf_ = q_dot_lpf_;
        }
        q_noise_pre_ = q_noise_;
    }
    time_pre_ = time_cur_;
}


void CustomController::processObservation() // [linvel, angvel, proj_grav, commands, dof_pos, dof_vel, actions]
{


    int data_idx = 0;

    Eigen::Quaterniond q;
    q.x() = rd_cc_.q_virtual_(3);
    q.y() = rd_cc_.q_virtual_(4);
    q.z() = rd_cc_.q_virtual_(5);
    q.w() = rd_cc_.q_virtual_(MODEL_DOF_QVIRTUAL-1);   
    
    base_lin_vel = q.conjugate()*(rd_cc_.q_dot_virtual_.segment(0,3));
    base_ang_vel = (rd_cc_.q_dot_virtual_.segment(3,3));

    for (int i = 0; i < 3; i++){
        state_cur_[data_idx] = base_ang_vel(i);
        data_idx++;
    }

    Vector3_t grav, projected_grav, forward_vec;
    grav << 0, 0, -1.;
    forward_vec << 1., 0, 0;
    projected_grav = q.conjugate()*grav;

    Vector3_t forward = q * forward_vec;
    double heading = atan2(forward(1), forward(0));
    double heading_error_ = target_heading_ - heading;

    state_cur_[data_idx] = projected_grav(0);
    data_idx++;
    state_cur_[data_idx] = projected_grav(1);
    data_idx++;
    state_cur_[data_idx] = projected_grav(2);
    data_idx++;

    float prev_step_period_ = step_period_;
    commands_(0) = 0.5;
    commands_(1) = 0.0;
    commands_(2) = 0.0;
    state_cur_[data_idx] = commands_(0);
    data_idx++;
    state_cur_[data_idx] = commands_(1);
    data_idx++;
    if (heading_mode_) commands_(2) = DyrosMath::minmax_cut(2*heading_error_, -1., 1.);
    state_cur_[data_idx] = commands_(2);
    // cout << "[DEBUG] Heading: " << commands_(2) << endl;
    data_idx++;
    step_period_ = DyrosMath::minmax_cut(min(max_stride_x/(abs(commands_(0))+1.e-6), min(max_stride_y/(abs(commands_(1))+1.e-6), max_stride_yaw/(abs(commands_(2))+1.e-6))), 0.4, 0.8);
    step_ticks_ *= step_period_ / prev_step_period_;
    for (int i = 0; i < num_actuator_action; i++)
    {
        state_cur_[data_idx] = q_noise_(i) - q_init_(i);
        data_idx++;
    }

    for (int i = 0; i < num_actuator_action; i++)
    {
        if (is_on_robot_)
        {
            state_cur_[data_idx] = q_vel_noise_(i);
        }
        else
        {
            state_cur_[data_idx] = q_vel_noise_(i); //rd_cc_.q_dot_virtual_(i+6);
        }
        data_idx++;
    }
    // std::cout << "step ticks : " << step_ticks_ << std::endl;
    // std::cout << "step period : " << step_period_ << std::endl;
    state_cur_[data_idx] = cos(2*M_PI*(step_ticks_+phase_indicator_*step_period_)/(2*step_period_));
    data_idx++;
    state_cur_[data_idx] = sin(2*M_PI*(step_ticks_+phase_indicator_*step_period_)/(2*step_period_));
    data_idx++;

    for (int i = 0; i <num_actuator_action; i++) 
    {
        state_cur_[data_idx] = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);
        data_idx++;
    }
    assert(data_idx == num_cur_state);
    for (int i = 0; i < num_cur_critic_state; i++){
        if (i < num_cur_state) critic_state_cur_[i] = state_cur_[i];
        else critic_state_cur_[i] = 0.;
    }
    std::copy(critic_state_cur_.begin(),
                critic_state_cur_.begin() + num_cur_critic_state,
                input_states_buffer_n[0].begin());

    // Run normalizer if available
    Ort::Session* normalizer_session = getSession("normalizer");
    if (normalizer_session) {
        output_tensors_n = normalizer_session->Run(Ort::RunOptions{nullptr}, input_names_char_n.data(), input_tensors_n.data(), input_number_n, output_names_char_n.data(), output_number_n);
        for (size_t i = 0; i < num_cur_critic_state; i++) {
            normalized_state_cur_[i] = output_tensors_n[0].GetTensorMutableData<float>()[i];
        }
    }

    std::copy(normalized_state_cur_.begin(),
                normalized_state_cur_.begin() + num_cur_state,
                input_states_buffer[input_obs_idx_].begin());
    std::copy(h_cur_.begin(),
                h_cur_.begin() + num_cur_h,
                input_states_buffer[input_h0_idx_].begin());

}

void CustomController::feedforwardPolicy()
{
    // Get actor session
    Ort::Session* actor_session = getSession("actor");
    if (!actor_session) {
        std::cerr << "[ERROR] Actor session not available!" << std::endl;
        return;
    }

    output_tensors = actor_session->Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(), input_number, output_names_char.data(), output_number);

    for (size_t i = 0; i < output_tensors.size(); i++) {
        if (!output_tensors[i].IsTensor()) {
            std::cerr << "Output " << i << " is not a valid tensor." << std::endl;
            continue;
        }
    }

    // output tensor to rl_action_
    for (size_t i = 0; i < num_actuator_action; i++) {
        rl_action_(i) = output_tensors[output_action_idx_].GetTensorMutableData<float>()[i];
    }

}

void CustomController::processEverythingElse()
{
    for (size_t i = 0; i < num_cur_h; i++){
        h_cur_[i] = output_tensors[output_hn_idx_].GetTensorMutableData<float>()[i];
    }
    // for (size_t i = 0; i < num_cur_latent; i++) {
    //     latent_cur_[i] = output_tensors[output_latent_idx_].GetTensorMutableData<float>()[i];
    // }
    // // cout << "RL Action: " << rl_action_.transpose() << endl;

    // std::copy(latent_cur_.begin(),
    //             latent_cur_.begin() + num_cur_latent,
    //             input_states_buffer_d[0].begin());
    // // output tensor to critic obs
    // output_tensors_d = session_d.Run(Ort::RunOptions{nullptr}, input_names_char_d.data(), input_tensors_d.data(), input_number_d, output_names_char_d.data(), output_number_d);

    // for (size_t i = 0; i < output_tensors_d.size(); i++) {
    //     if (!output_tensors_d[i].IsTensor()) {
    //         std::cerr << "Decoder output " << i << " is not a valid tensor." << std::endl;
    //         continue;
    //     }
    // }

    // for (size_t i = 0; i < num_cur_critic_state; i++) {
    //     normalized_critic_state_cur_[i] = output_tensors_d[0].GetTensorMutableData<float>()[i];
    // }

    // std::copy(normalized_critic_state_cur_.begin(),
    //             normalized_critic_state_cur_.begin() + num_cur_critic_state,
    //             input_states_buffer_c[0].begin());
    // std::copy(normalized_critic_state_cur_.begin(),
    //             normalized_critic_state_cur_.begin() + num_cur_critic_state,
    //             input_states_buffer_dn[0].begin());
    // // output tensor to value_
    // output_tensors_c = session_c.Run(Ort::RunOptions{nullptr}, input_names_char_c.data(), input_tensors_c.data(), input_number_c, output_names_char_c.data(), output_number_c);
    // output_tensors_dn = session_dn.Run(Ort::RunOptions{nullptr}, input_names_char_dn.data(), input_tensors_dn.data(), input_number_dn, output_names_char_dn.data(), output_number_dn);
    // value_ = output_tensors_c[0].GetTensorMutableData<float>()[0];
    // for (size_t i = 0; i < num_cur_critic_state; i++) {
    //     critic_state_cur_[i] = output_tensors_dn[0].GetTensorMutableData<float>()[i];
    // }
    // std::cout << "value : " << value_ << std::endl;
    // int data_idx = 0;
    // data_idx += num_cur_state;
    // std::cout << "predicted lin vel : " << critic_state_cur_[data_idx] << "\t" << critic_state_cur_[data_idx+1] << "\t" << critic_state_cur_[data_idx+2] << std::endl;
    // data_idx += 3;
    // data_idx += 8;
    // std::cout << "predicted reward : " << critic_state_cur_[data_idx] << std::endl;
    // data_idx += 1;
    // std::cout << "predicted z contact indicator : " << critic_state_cur_[data_idx] << "\t" << critic_state_cur_[data_idx+1] << std::endl;
    // data_idx += 2;
    // std::cout << "predicted injected torque : " ;
    // for (int i = 0; i < 12; i++)
    //    std::cout << critic_state_cur_[data_idx + i] << "\t";
    // std::cout << std::endl;
    // data_idx += 12;
    // std::cout << "predicted injected force : ";
    // for (int i = 0; i < 3; i++)
    //    std::cout << critic_state_cur_[data_idx + i] << "\t";
    // std::cout << std::endl;



    if (is_write_file_)
    {
            writeFile << (rd_cc_.control_time_us_ - time_inference_pre_)/1e6 << "\t";
            writeFile << rd_cc_.LF_CF_FT.transpose() << "\t";
            writeFile << rd_cc_.RF_CF_FT.transpose() << "\t";

            writeFile << rd_cc_.torque_desired.transpose()  << "\t";
            writeFile << q_noise_.transpose() << "\t";
            writeFile << q_dot_lpf_.transpose() << "\t";
            writeFile << base_lin_vel.transpose() << "\t" << base_ang_vel.transpose() << "\t" << rd_cc_.q_dot_virtual_.segment(6,33).transpose() << "\t";
            writeFile << rd_cc_.q_virtual_.transpose() << "\t";
            writeFile << heading << "\t";

            writeFile << value_ << "\t" << stop_by_value_thres_ << "\t";
            writeFile << commands_(0) << "\t" << commands_(1) << "\t" << commands_(2) << "\t";
            
            // Right foot global position (x, y, z)
            writeFile << rd_cc_.link_[Right_Foot].xpos(0) << "\t" 
                      << rd_cc_.link_[Right_Foot].xpos(1) << "\t" 
                      << rd_cc_.link_[Right_Foot].xpos(2) << "\t";
            
            // Left foot global position (x, y, z)
            writeFile << rd_cc_.link_[Left_Foot].xpos(0) << "\t" 
                      << rd_cc_.link_[Left_Foot].xpos(1) << "\t" 
                      << rd_cc_.link_[Left_Foot].xpos(2) << "\t";
            
            writeFile << std::endl;
            time_write_pre_ = rd_cc_.control_time_us_;
        }
    time_inference_pre_ = rd_cc_.control_time_us_;

}

void CustomController::computeSlow()

{

    copyRobotData(rd_);

    if (rd_cc_.tc_.mode == 7)

    {

        if (rd_cc_.tc_init)

        {

            //Initialize settings for Task Control! 

            start_time_ = rd_cc_.control_time_us_;

            q_noise_pre_ = q_noise_ = q_init_ = rd_cc_.q_virtual_.segment(6,MODEL_DOF);

            q_leg_desired_ = rd_cc_.q_.segment(0,12);

            time_cur_ = start_time_ / 1e6;

            time_pre_ = time_cur_ - 0.005;

            // time_inference_pre_ = rd_cc_.control_time_us_ - (1/249.9)*1e6;

            time_inference_pre_ = rd_cc_.control_time_us_ - (1/(hz_))*1e6;

            rd_.tc_init = false;

            std::cout<<"cc mode 7"<<std::endl;

            torque_init_ = rd_cc_.torque_desired;

            processNoise();

            processBias();

            processObservation();
        }

        processNoise();

        processBias();

        if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= 1/hz_) // 125 is the control frequency

        {

            processObservation();

            feedforwardPolicy();
            
            updateNextStepTime();

            action_dt_accumulate_ += DyrosMath::minmax_cut(rl_action_(num_action-1)*5/hz_, 0.0, 5/hz_);

            if (value_ < 0.)
            {
                if (stop_by_value_thres_ == false)
                {
                    stop_by_value_thres_ = true;
                    stop_start_time_ = rd_cc_.control_time_us_;
                    q_stop_ = q_noise_;
                    std::cout << "Stop by Value Function : " << walking_tick << ", Value : " << value_ << std::endl;
                }
            }

        }

        for (int i = 0; i < num_actuator_action; i++){
            if (ctrl_type_ == "T"){
                torque_rl_(i) = DyrosMath::minmax_cut(rl_action_(i), -1., 1.) *torque_bound_(i) ;
            }
            if (ctrl_type_ == "P"){
                // Use pre-computed values for faster execution
                torque_rl_(i) = DyrosMath::minmax_cut(kp_(i,i) * (DyrosMath::minmax_cut(rl_action_(i), -1., 1.) * pd_limit_std_(i) + pd_limit_bias_(i) - q_noise_(i)) - kv_(i,i)*q_vel_noise_(i), -torque_bound_(i), torque_bound_(i));
            }

        }
        
        for (int i = num_actuator_action; i < MODEL_DOF; i++)
            torque_rl_(i) = kp_(i,i) * (q_init_(i) - q_noise_(i)) - kv_(i,i)*q_vel_noise_(i);
        
        if (rd_cc_.control_time_us_ < start_time_ + 0.1e6)
        {
            for (int i = 0; i <MODEL_DOF; i++)
                torque_spline_(i) = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 0.1e6, torque_init_(i), torque_rl_(i), 0.0, 0.0);

            rd_.torque_desired = torque_spline_;    
        }
        else
             rd_.torque_desired = torque_rl_;

        if (stop_by_value_thres_)
            rd_.torque_desired = kp_ * (q_stop_ - q_noise_) - kv_*q_vel_noise_;
        if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= 1/hz_) // 125 is the control frequency
            processEverythingElse();


    }

}
void CustomController::computeFast(){}

void CustomController::computePlanner(){}

void CustomController::copyRobotData(RobotData &rd_l)
{
    std::memcpy(&rd_cc_, &rd_l, sizeof(RobotData));
}

void CustomController::loadCommand(const std::string &command_file)
{
    std::ifstream file(command_file);
    if (!file)
    {
    throw std::runtime_error("Cannot open command file: " + command_file);
    }

    std::string line;
    while (std::getline(file, line))
    {
    if (line.empty())
    continue;

    std::istringstream iss(line);
    std::string keyval;
    iss >> keyval;

    auto eq_pos = keyval.find('=');
    if (eq_pos == std::string::npos)
        throw std::runtime_error("Expected '=' in line: " + keyval);

    std::string key = keyval.substr(0, eq_pos);
    float vec = std::stof(keyval.substr(eq_pos + 1));

    if (key == "target_vel_x")
    commands_(0) = vec;
    else if (key == "target_vel_y")
    commands_(1) = vec;
    else if (key == "target_vel_yaw")
    commands_(2) = vec;
    else if (key == "target_heading")
    target_heading_ = vec;
    else
    std::cerr << "Warning: Unknown key '" << key << "' in file " << command_file << std::endl;
    }
    file.close();
}

void CustomController::updateNextStepTime()
{           
    step_ticks_ += del_t;
    if (step_ticks_ >= step_period_) {
        step_ticks_ = 0.;
        phase_indicator_ = 1-phase_indicator_;
    }
}

void CustomController::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
    commands_(0) = DyrosMath::minmax_cut(vel_scale_x_*joy->axes[1], -0.5, 1.0);
    commands_(1) = DyrosMath::minmax_cut(vel_scale_y_*joy->axes[0] , -0.8, 0.8);

    if (joy->buttons[1] == 1.0 && vel_scale_x_ < 1.0 && vel_scale_y_ < 0.3){
        vel_scale_x_ += 0.03;
        vel_scale_y_ += 0.01;
        ROS_INFO("Velocity X : %f", vel_scale_x_);
        ROS_INFO("Velocity Y : %f", vel_scale_y_);
    }

    if (joy->buttons[0] == 1.0 && vel_scale_x_ > 0.1 && vel_scale_y_ > 0.03){
        vel_scale_x_ -= 0.03;
        vel_scale_y_ -= 0.01;
        ROS_INFO("Velocity X : %f", vel_scale_x_);
        ROS_INFO("Velocity Y : %f", vel_scale_y_);
    }
    if(joy->buttons[6] == 1){
        commands_(2) = 0.6;
    }
    if(joy->buttons[7] == 1){
        commands_(2) = -0.6;
    }
    if(joy->buttons[6] != 1 && joy->buttons[7] != 1){
        commands_(2) = 0.;
    }
}


Eigen::VectorQd CustomController::getControl()
{
    return ControlVal_;
}