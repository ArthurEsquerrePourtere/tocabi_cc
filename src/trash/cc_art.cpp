#include "cc.h"
#include <algorithm>
#include <iomanip>
#include <cmath>

using namespace TOCABI;

namespace
{
constexpr int IDX_WAIST1 = 12;
constexpr int IDX_WAIST2 = 13;
constexpr int IDX_UPPERBODY = 14;
constexpr int IDX_L_SHOULDER1 = 15;
constexpr int IDX_L_SHOULDER2 = 16;
constexpr int IDX_L_ELBOW = 19;
constexpr int IDX_R_SHOULDER1 = 25;
constexpr int IDX_R_SHOULDER2 = 26;
constexpr int IDX_R_ELBOW = 29;
constexpr double TWO_PI = 6.28318530717958647692;
constexpr double HALF_PI = 1.57079632679489661923;
constexpr double AXIS_SIGN_L_SHOULDER2 = 1.0;
constexpr double AXIS_SIGN_R_SHOULDER2 = -1.0;
constexpr double AXIS_SIGN_L_ELBOW = 1.0;
constexpr double AXIS_SIGN_R_ELBOW = 1.0;
}

CustomController::CustomController(RobotData &rd) : rd_(rd), //, wbc_(dc.wbc_)
        env(ORT_LOGGING_LEVEL_WARNING, "tocabi"),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        session(nullptr),
        session_n(nullptr),
        session_d(nullptr),
        session_c(nullptr),
        session_dn(nullptr)
{
    ControlVal_.setZero();

    if (is_write_file_)
    {
        if (is_on_robot_)
        {
            writeFile.open("/home/dyros/catkin_ws/src/tocabi_cc/result/data.csv", std::ofstream::out);
        }
        else
        {
            writeFile.open("/home/dyros/catkin_ws/src/tocabi_cc/result/art_data.csv", std::ofstream::out);
        }
        writeFile << std::fixed << std::setprecision(8);
        
        // Write CSV header
        writeFile << "time\t"
                  << "LF_FT_fx\tLF_FT_fy\tLF_FT_fz\tLF_FT_tx\tLF_FT_ty\tLF_FT_tz\t"
                  << "RF_FT_fx\tRF_FT_fy\tRF_FT_fz\tRF_FT_tx\tRF_FT_ty\tRF_FT_tz\t"
                  << "torque_des_0\ttorque_des_1\ttorque_des_2\ttorque_des_3\ttorque_des_4\ttorque_des_5\t"
                  << "torque_des_6\ttorque_des_7\ttorque_des_8\ttorque_des_9\ttorque_des_10\ttorque_des_11\t"
                  << "q_0\tq_1\tq_2\tq_3\tq_4\tq_5\tq_6\tq_7\tq_8\tq_9\tq_10\tq_11\t"
                  << "qdot_lpf_0\tqdot_lpf_1\tqdot_lpf_2\tqdot_lpf_3\tqdot_lpf_4\tqdot_lpf_5\t"
                  << "qdot_lpf_6\tqdot_lpf_7\tqdot_lpf_8\tqdot_lpf_9\tqdot_lpf_10\tqdot_lpf_11\t"
                  << "base_lin_vel_x\tbase_lin_vel_y\tbase_lin_vel_z\t"
                  << "base_ang_vel_x\tbase_ang_vel_y\tbase_ang_vel_z\t"
                  << "cmd_x\tcmd_y\tcmd_yaw\t"
                  << "rf_x\trf_y\trf_z\t"
                  << "lf_x\tlf_y\tlf_z\t"
                  << "base_height\t"
                  << std::endl;
    }
    initVariable();
    loadOnnX();

    joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy", 10, &CustomController::joyCallback, this);
    
    // Initialize target velocity publisher for MuJoCo visualization
    target_vel_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/mujoco_ros_interface/target_velocity", 10);
    ROS_INFO("Target velocity publisher initialized on topic: /mujoco_ros_interface/target_velocity");
}

void CustomController::initVariable()
{    
    // Load the path from the configuration file

    rl_action_.resize(num_action, 1);

    state_cur_.resize(num_cur_state, 1);
    critic_state_cur_.resize(num_cur_critic_state, 1);
    normalized_state_cur_.resize(num_cur_state, 1);
    normalized_critic_state_cur_.resize(num_cur_critic_state, 1);
    h_cur_.resize(num_cur_h, 1);
    latent_cur_.resize(num_cur_latent, 1);
    std::fill(h_cur_.begin(), h_cur_.end(), 0.0f);

    q_dot_lpf_.setZero();

    torque_bound_ << 333, 232, 263, 289, 222, 166,
                    333, 232, 263, 289, 222, 166,
                    303, 303, 303, 
                    64, 64, 64, 64, 23, 23, 10, 10,
                    10, 10,
                    64, 64, 64, 64, 23, 23, 10, 10;  
            
                    
    q_init_ << 0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0.0, 0.0, 0.0,
                0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0,
                0.0, 0.0,
                -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0;


    kp_.setZero();
    kv_.setZero();
    kp_.diagonal() <<   2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
                        2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
                        6000.0, 10000.0, 10000.0,
                        400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0,
                        100.0, 100.0,
                        400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0;
    kp_.diagonal() /= 9.0;
    // kp_.diagonal() *= 4.0;
    kv_.diagonal() << 15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        200.0, 100.0, 100.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
                        2.0, 2.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0;
    kv_.diagonal() /= 3.0;
    // kv_.diagonal() *= 2.0;

    pd_limit(0, 0) = -0.6;
    pd_limit(0, 1) = 0.8;
    pd_limit(1, 0) = -1.;
    pd_limit(1, 1) = 1.;
    pd_limit(2, 0) = -1.;
    pd_limit(2, 1) = 1.;
    pd_limit(3, 0) = -0.3;
    pd_limit(3, 1) = 1.5;
    pd_limit(4, 0) = -1.;
    pd_limit(4, 1) = 1.;
    pd_limit(5, 0) = -1.;
    pd_limit(5, 1) = 1.;
    pd_limit(6, 0) = -0.8;
    pd_limit(6, 1) = 0.6;
    pd_limit(7, 0) = -1.;
    pd_limit(7, 1) = 1.;
    pd_limit(8, 0) = -1.;
    pd_limit(8, 1) = 1.;
    pd_limit(9, 0) = -0.3;
    pd_limit(9, 1) = 1.5;
    pd_limit(10, 0) = -1.;
    pd_limit(10, 1) = 1.;
    pd_limit(11, 0) = -1.;
    pd_limit(11, 1) = 1.;

    // Woohyun
    initBias();
    base_lin_vel.setZero();
    base_ang_vel.setZero();
    base_ang_vel_lpf_.setZero();
    command_vel_filtered_.setZero();
    command_vel_filtered_prev_.setZero();
    arm_swing_phase_ = 0.0;


}

void CustomController::loadOnnX()
{
    string cur_path = "/home/dyros/raibertGRU_ws/src/tocabi_cc/";
    string actor_path = cur_path + "onnx_files/actor.onnx";
    string normalizer_path = cur_path + "onnx_files/normalizer.onnx";
    string denormalizer_path = cur_path + "onnx_files/denormalizer.onnx";
    string decoder_path = cur_path + "onnx_files/decoder.onnx";
    string critic_path = cur_path + "onnx_files/critic.onnx";
    if (is_on_robot_)
    {
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/";
        actor_path = cur_path + "onnx_files/actor.onnx"; 
        normalizer_path = cur_path + "onnx_files/normalizer.onnx";
        denormalizer_path = cur_path + "onnx_files/denormalizer.onnx";
        decoder_path = cur_path + "onnx_files/decoder.onnx";
        critic_path = cur_path + "onnx_files/critic.onnx";
    }

    if (ctrl_mode == 1){
        loadCommand(cur_path + "commands.txt");
    }


    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.AddConfigEntry("session.use_deterministic_compute", "1");

    session = Ort::Session(env, actor_path.c_str(), session_options);
    session_n = Ort::Session(env, normalizer_path.c_str(), session_options);
    session_dn = Ort::Session(env, denormalizer_path.c_str(), session_options);
    session_d = Ort::Session(env, decoder_path.c_str(), session_options);
    session_c = Ort::Session(env, critic_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    input_number = session.GetInputCount();
    output_number = session.GetOutputCount();
    input_number_n = session_n.GetInputCount();
    output_number_n = session_n.GetOutputCount();
    input_number_dn = session_dn.GetInputCount();
    output_number_dn = session_dn.GetOutputCount();
    input_number_d = session_d.GetInputCount();
    output_number_d = session_d.GetOutputCount();
    input_number_c = session_c.GetInputCount();
    output_number_c = session_c.GetOutputCount();

    input_names.resize(input_number);
    output_names.resize(output_number);
    input_names_n.resize(input_number_n);
    output_names_n.resize(output_number_n);
    input_names_dn.resize(input_number_dn);
    output_names_dn.resize(output_number_dn);
    input_names_d.resize(input_number_d);
    output_names_d.resize(output_number_d);
    input_names_c.resize(input_number_c);
    output_names_c.resize(output_number_c);

    input_names_char.resize(input_names.size());
    output_names_char.resize(output_names.size());
    input_names_char_n.resize(input_names_n.size());
    output_names_char_n.resize(output_names_n.size());
    input_names_char_dn.resize(input_names_dn.size());
    output_names_char_dn.resize(output_names_dn.size());
    input_names_char_d.resize(input_names_d.size());
    output_names_char_d.resize(output_names_d.size());
    input_names_char_c.resize(input_names_c.size());
    output_names_char_c.resize(output_names_c.size());

    for (size_t i = 0; i < input_number; i++) {
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
        input_names[i] = input_name.get();
    }
    for (size_t i = 0; i < output_number; i++) {
        Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
        output_names[i] = output_name.get();
    }

    for (size_t i = 0; i < input_number_n; i++) {
        Ort::AllocatedStringPtr input_name_n = session_n.GetInputNameAllocated(i, allocator);
        input_names_n[i] = input_name_n.get();
    }
    for (size_t i = 0; i < output_number_n; i++) {
        Ort::AllocatedStringPtr output_name_n = session_n.GetOutputNameAllocated(i, allocator);
        output_names_n[i] = output_name_n.get();
    }

    for (size_t i = 0; i < input_number_dn; i++) {
        Ort::AllocatedStringPtr input_name_dn = session_dn.GetInputNameAllocated(i, allocator);
        input_names_dn[i] = input_name_dn.get();
    }
    for (size_t i = 0; i < output_number_dn; i++) {
        Ort::AllocatedStringPtr output_name_dn = session_dn.GetOutputNameAllocated(i, allocator);
        output_names_dn[i] = output_name_dn.get();
    }

    for (size_t i = 0; i < input_number_d; i++) {
        Ort::AllocatedStringPtr input_name_d = session_d.GetInputNameAllocated(i, allocator);
        input_names_d[i] = input_name_d.get();
    }
    for (size_t i = 0; i < output_number_d; i++) {
        Ort::AllocatedStringPtr output_name_d = session_d.GetOutputNameAllocated(i, allocator);
        output_names_d[i] = output_name_d.get();
    }

    for (size_t i = 0; i < input_number_c; i++) {
        Ort::AllocatedStringPtr input_name_c = session_c.GetInputNameAllocated(i, allocator);
        input_names_c[i] = input_name_c.get();
    }
    for (size_t i = 0; i < output_number_c; i++) {
        Ort::AllocatedStringPtr output_name_c = session_c.GetOutputNameAllocated(i, allocator);
        output_names_c[i] = output_name_c.get();
    }

    // Print input/output names
    std::cout << "Input names: "; 
    std::copy(input_names.begin(), input_names.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;

    std::cout << "Output names: ";
    std::copy(output_names.begin(), output_names.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;
    
    // Print input/output names
    std::cout << "Input names Normalizer: "; 
    std::copy(input_names_n.begin(), input_names_n.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;

    std::cout << "Output names Normalizer: ";
    std::copy(output_names_n.begin(), output_names_n.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;

    // Print input/output names
    std::cout << "Input names Normalizer: "; 
    std::copy(input_names_dn.begin(), input_names_dn.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;

    std::cout << "Output names Normalizer: ";
    std::copy(output_names_dn.begin(), output_names_dn.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;

    // Print input/output names
    std::cout << "Input names Decoder: "; 
    std::copy(input_names_d.begin(), input_names_d.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;

    std::cout << "Output names Decoder: ";
    std::copy(output_names_d.begin(), output_names_d.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;

    // Print input/output names
    std::cout << "Input names Critic: "; 
    std::copy(input_names_c.begin(), input_names_c.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;

    std::cout << "Output names Critic: ";
    std::copy(output_names_c.begin(), output_names_c.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;

    for (size_t i = 0; i < input_names.size(); ++i) { 
        input_names_char[i] = input_names[i].c_str();
        if (input_names_char[i] == "obs") {input_obs_idx_ = i;}
        else if (input_names_char[i] == "h0"){input_h0_idx_ = i;}
    }
    for (size_t i = 0; i < output_names.size(); ++i) { 
        output_names_char[i] = output_names[i].c_str();
        if (output_names_char[i] == "action") {output_action_idx_ = i;}
        else if (output_names_char[i] == "latent") {output_latent_idx_ = i;}
        else if (output_names_char[i] == "hn"){output_hn_idx_ = i;}
    }
    // Initialize input tensors
    for (size_t i = 0; i < input_number; ++i) {
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = tensor_info.GetShape();
        cout << "Input " << i << " shape: " << input_shape.size() << endl;
        std::vector<float> input_tensor_values(tensor_info.GetElementCount(), 0.0);
        input_states_buffer.push_back(std::move(input_tensor_values));

        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            input_states_buffer.back().data(),
            input_states_buffer.back().size(),
            input_shape.data(),
            input_shape.size()));
    }

    for (size_t i = 0; i < input_names_n.size(); ++i) { 
        input_names_char_n[i] = input_names_n[i].c_str();
    }
    for (size_t i = 0; i < output_names_n.size(); ++i) { 
        output_names_char_n[i] = output_names_n[i].c_str();
    }
    // Initialize input tensors
    for (size_t i = 0; i < input_number_n; ++i) {
        Ort::TypeInfo type_info = session_n.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape_n = tensor_info.GetShape();
        cout << "Normalizer Input " << i << " shape: " << input_shape_n.size() << endl;
        std::vector<float> input_tensor_values(tensor_info.GetElementCount(), 0.0);
        input_states_buffer_n.push_back(std::move(input_tensor_values));

        input_tensors_n.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            input_states_buffer_n.back().data(),
            input_states_buffer_n.back().size(),
            input_shape_n.data(),
            input_shape_n.size()));
    }

    for (size_t i = 0; i < input_names_dn.size(); ++i) { 
        input_names_char_dn[i] = input_names_dn[i].c_str();
    }
    for (size_t i = 0; i < output_names_dn.size(); ++i) { 
        output_names_char_dn[i] = output_names_dn[i].c_str();
    }
    // Initialize input tensors
    for (size_t i = 0; i < input_number_dn; ++i) {
        Ort::TypeInfo type_info = session_dn.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape_dn = tensor_info.GetShape();
        cout << "Normalizer Input " << i << " shape: " << input_shape_dn.size() << endl;
        std::vector<float> input_tensor_values(tensor_info.GetElementCount(), 0.0);
        input_states_buffer_dn.push_back(std::move(input_tensor_values));

        input_tensors_dn.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            input_states_buffer_dn.back().data(),
            input_states_buffer_dn.back().size(),
            input_shape_dn.data(),
            input_shape_dn.size()));
    }

    for (size_t i = 0; i < input_names_d.size(); ++i) { 
        input_names_char_d[i] = input_names_d[i].c_str();
    }
    for (size_t i = 0; i < output_names_d.size(); ++i) { 
        output_names_char_d[i] = output_names_d[i].c_str();
    }
    // Initialize input tensors
    for (size_t i = 0; i < input_number_d; ++i) {
        Ort::TypeInfo type_info = session_d.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape_d = tensor_info.GetShape();
        cout << "Decoder Input " << i << " shape: " << input_shape_d.size() << endl;
        std::vector<float> input_tensor_values(tensor_info.GetElementCount(), 0.0);
        input_states_buffer_d.push_back(std::move(input_tensor_values));

        input_tensors_d.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            input_states_buffer_d.back().data(),
            input_states_buffer_d.back().size(),
            input_shape_d.data(),
            input_shape_d.size()));
    }

    for (size_t i = 0; i < input_names_c.size(); ++i) { 
        input_names_char_c[i] = input_names_c[i].c_str();
    }
    for (size_t i = 0; i < output_names_c.size(); ++i) { 
        output_names_char_c[i] = output_names_c[i].c_str();
    }
    // Initialize input tensors
    for (size_t i = 0; i < input_number_c; ++i) {
        Ort::TypeInfo type_info = session_c.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape_c = tensor_info.GetShape();
        cout << "Critic Input " << i << " shape: " << input_shape_c.size() << endl;
        std::vector<float> input_tensor_values(tensor_info.GetElementCount(), 0.0);
        input_states_buffer_c.push_back(std::move(input_tensor_values));

        input_tensors_c.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            input_states_buffer_c.back().data(),
            input_states_buffer_c.back().size(),
            input_shape_c.data(),
            input_shape_c.size()));
    }
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
            // only apply the filter to the first 2 values of q_noise
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), leg_dof_vel_lpf_cutoff_freq_);
            q_dot_lpf_higher_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_higher_, 1/(time_cur_ - time_pre_), feet_dof_vel_lpf_cutoff_freq_);
            // revert all values except first 2 values to unfiltered
            for (int i = 4; i < 6; i++) {
                q_dot_lpf_(i) = q_dot_lpf_higher_(i);
            }
            for (int i = 10; i < MODEL_DOF; i++) {
                q_dot_lpf_(i) = q_dot_lpf_higher_(i);
            }
            // q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 20.0);

            // cout << "q_noise_: " << q_noise_.transpose().head<12>() << endl;
            // cout << "q_noise_pre_: " << q_noise_pre_.transpose().head<12>() << endl;
            // cout << "time_cur_: " << time_cur_ << ", time_pre_: " << time_pre_ << endl;
            // cout << "diff q: " << (q_noise_ - q_noise_pre_).transpose().head<12>() << endl;
            // cout << "diff time: " << (time_cur_ - time_pre_) << endl;
            // cout << "q_vel_noise_: " << q_vel_noise_.transpose().head<12>() << endl;
            // cout << "q_dot_lpf_: " << q_dot_lpf_.transpose().head<12>() << endl;
            base_ang_vel = (rd_cc_.q_dot_virtual_.segment(3,3));
            base_ang_vel_lpf_ = DyrosMath::lpf<3>(base_ang_vel, base_ang_vel_lpf_, 1/(time_cur_ - time_pre_), ang_vel_lpf_cutoff_freq_);
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

    // cout << "base_ang_vel: " << base_ang_vel.transpose() << endl;
    // cout << "base_ang_vel_lpf_: " << base_ang_vel_lpf_.transpose() << endl;

    for (int i = 0; i < 3; i++){
        if (use_lpf_vel_obs_){
            state_cur_[data_idx] = base_ang_vel_lpf_(i);
        }
        else{
            state_cur_[data_idx] = base_ang_vel(i);
        }
        // state_cur_[data_idx] = base_ang_vel_lpf_(i);
        // state_cur_[data_idx] = base_ang_vel(i);
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
    if (ctrl_mode == 2 and !is_on_robot_){
        string cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/";
        updateCommandFromTimeline(cur_path + "timeline.txt");
    }
    // commands_(0) = 0.0;
    // commands_(1) = 0.5;
    // commands_(2) = 0.;
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
            if (use_lpf_vel_obs_){
                state_cur_[data_idx] = q_dot_lpf_(i);
            }
            else{
                state_cur_[data_idx] = q_vel_noise_(i); //rd_cc_.q_dot_virtual_(i+6);
            }
            // state_cur_[data_idx] = q_dot_lpf_(i);
            // state_cur_[data_idx] = rd_cc_.q_dot_virtual_(i+6);
            // if ((i < 4) || (i >= 6 && i < 10))
            //     state_cur_[data_idx] = rd_cc_.q_dot_virtual_(i+6)/2.5;
            // else
            //     state_cur_[data_idx] = rd_cc_.q_dot_virtual_(i+6)/2.;
            //clip the value to be between -5 and 5
            // state_cur_[data_idx] = DyrosMath::minmax_cut(rd_cc_.q_dot_virtual_(i+6), -0.3, 0.3);
            // check if abs value is less than 2
            // if (abs(rd_cc_.q_dot_virtual_(i+6)) < 1.0) {
            //     state_cur_[data_idx] = rd_cc_.q_dot_virtual_(i+6);
            // }
            // else {
            //     if (rd_cc_.q_dot_virtual_(i+6) > 0) {
            //         state_cur_[data_idx] = 1.0 + 0.2*(rd_cc_.q_dot_virtual_(i+6) - 1.0);
            //     }
            //     else {
            //         state_cur_[data_idx] = -1.0 + 0.2*(rd_cc_.q_dot_virtual_(i+6) + 1.0);
            //     }
            // }
        }
        data_idx++;
    }
    // if (value_ < 3.2){
    //     std::cout << "phase indicator : " << phase_indicator_ << std::endl;
    //     std::cout << "step ticks : " << step_ticks_ << std::endl;
    //     std::cout << "step period : " << step_period_ << std::endl;
    // }
    state_cur_[data_idx] = cos(2*M_PI*(step_ticks_+phase_indicator_*step_period_)/(2*step_period_));
    // if (value_ < 3.2){
    //     cout << "cosine value: " << state_cur_[data_idx] << endl;
    // }
    data_idx++;
    state_cur_[data_idx] = sin(2*M_PI*(step_ticks_+phase_indicator_*step_period_)/(2*step_period_));
    // if (value_ < 3.2){
    //     cout << "sine value: " << state_cur_[data_idx] << endl;
    // }
    data_idx++;

    for (int i = 0; i <num_actuator_action; i++) 
    {   
        // cout << "rl_action_(" << i << "): " << rl_action_(i) << endl;

        state_cur_[data_idx] = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);
        data_idx++;
    }
    // cout << "torque init: " << torque_init_.transpose() << endl;
    // cout << "rl_action_: " << rl_action_.transpose() << endl;
    // exit(0);
    assert(data_idx == num_cur_state);
    for (int i = 0; i < num_cur_critic_state; i++){
        if (i < num_cur_state) critic_state_cur_[i] = state_cur_[i];
        else critic_state_cur_[i] = 0.;
    }
    std::copy(critic_state_cur_.begin(),
                critic_state_cur_.begin() + num_cur_critic_state,
                input_states_buffer_n[0].begin());

    output_tensors_n = session_n.Run(Ort::RunOptions{nullptr}, input_names_char_n.data(), input_tensors_n.data(), input_number_n, output_names_char_n.data(), output_number_n);
    for (size_t i = 0; i < num_cur_critic_state; i++) {
        normalized_state_cur_[i] = output_tensors_n[0].GetTensorMutableData<float>()[i];
    }

    std::copy(normalized_state_cur_.begin(),
                normalized_state_cur_.begin() + num_cur_state,
                input_states_buffer[input_obs_idx_].begin());
    std::copy(h_cur_.begin(),
                h_cur_.begin() + num_cur_h,
                input_states_buffer[input_h0_idx_].begin());

    // loop over observations and cout if something is over 10 or under -10
    // for (int i = 0; i < num_cur_state; i++){
    //     // cout << "Observation " << i << ": " << state_cur_[i] << endl;
    //     if (abs(state_cur_[i]) > 3.0){
    //         cout << "Observation " << i << " is out of bounds: " << state_cur_[i] << endl;
    //         exit(0);
    //     }
        
    // }
                
}

void CustomController::feedforwardPolicy()
{
    // cout << "Commands: " << commands_.transpose() << endl;
    // cout << "Commands to NN: " << endl;
    // float* float_ptr = input_tensors[0].GetTensorMutableData<float>();
    // std::cout << std::fixed << std::setprecision(3); // Keep decimal points steady
    // for (int i = 6; i < 9; i++) {
    //     std::cout << std::setw(8) << float_ptr[i]; 
    // }
    // std::cout << "\n"; 

    //print observations to debug, must print each type of observations seperately for a clear debugging, for instance for loop from 0 to 2 for ang vel, 3 to 5 for projected gravity, etc.
    // cout << "Observations to NN: " << endl;
    // float* float_ptr = input_tensors[0].GetTensorMutableData<float>();
    // std::cout << std::fixed << std::setprecision(5); // Keep decimal points steady
    // cout << "Ang Vel: ";
    // for (int i = 0; i < 3; i++) {
    //     std::cout << std::setw(8) << float_ptr[i]; 
    // }
    // std::cout << "\n";
    // cout << "Projected Gravity: ";
    // for (int i = 3; i < 6; i++) {
    //     std::cout << std::setw(8) << float_ptr[i]; 
    // }
    // std::cout << "\n";
    // cout << "Commands: ";
    // for (int i = 6; i < 9; i++) {
    //     std::cout << std::setw(8) << float_ptr[i]; 
    // }
    // std::cout << "\n";
    // cout << "Dof Pos: ";
    // for (int i = 9; i < 9 + num_actuator_action; i++) {
    //     std::cout << std::setw(8) << float_ptr[i]; 
    // }
    // std::cout << "\n";
    // cout << "Dof Vel: ";
    // for (int i = 9 + num_actuator_action; i < 9 + 2 * num_actuator_action; i++) {
    //     std::cout << std::setw(8) << float_ptr[i]; 
    // }
    // std::cout << "\n";
    // cout << "Phase Cosine & Sine: ";
    // for (int i = 9 + 2 * num_actuator_action; i < 9 + 2 * num_actuator_action + 2; i++) {
    //     std::cout << std::setw(8) << float_ptr[i]; 
    // }
    // std::cout << "\n";
    // cout << "Previous Actions: ";
    // for (int i = 9 + 2 * num_actuator_action + 2; i < 9 + 3 * num_actuator_action + 2; i++) {
    //     std::cout << std::setw(8) << float_ptr[i]; 
    // }
    // std::cout << "\n";

    output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(), input_number, output_names_char.data(), output_number);

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
    // if (value_ < 3.6)
    //     cout << "RL Action: " << rl_action_.transpose() << endl;

}

void CustomController::processEverythingElse()
{
    for (size_t i = 0; i < num_cur_h; i++){
        h_cur_[i] = output_tensors[output_hn_idx_].GetTensorMutableData<float>()[i];
    }
    for (size_t i = 0; i < num_cur_latent; i++) {
        latent_cur_[i] = output_tensors[output_latent_idx_].GetTensorMutableData<float>()[i];
    }
    // cout << "RL Action: " << rl_action_.transpose() << endl;

    std::copy(latent_cur_.begin(),
                latent_cur_.begin() + num_cur_latent,
                input_states_buffer_d[0].begin());
    // output tensor to critic obs
    output_tensors_d = session_d.Run(Ort::RunOptions{nullptr}, input_names_char_d.data(), input_tensors_d.data(), input_number_d, output_names_char_d.data(), output_number_d);

    for (size_t i = 0; i < output_tensors_d.size(); i++) {
        if (!output_tensors_d[i].IsTensor()) {
            std::cerr << "Decoder output " << i << " is not a valid tensor." << std::endl;
            continue;
        }
    }

    for (size_t i = 0; i < num_cur_critic_state; i++) {
        normalized_critic_state_cur_[i] = output_tensors_d[0].GetTensorMutableData<float>()[i];
    }

    // Debug: Compare normalized observation vs decoder output + Sim2Real gap analysis
    static int debug_counter = 0;
    static std::ofstream sim2real_debug_file;
    static bool file_opened = false;
    
    // Open file for sim2real analysis on first call
    if (!file_opened) {
        std::string debug_file_path;
        if (is_on_robot_) {
            debug_file_path = "/home/dyros/catkin_ws/src/tocabi_cc/result/sim2real_debug_real.csv";
        } else {
            debug_file_path = "/home/dyros/catkin_ws/src/tocabi_cc/result/sim2real_debug_sim.csv";
        }
        sim2real_debug_file.open(debug_file_path, std::ofstream::out);
        sim2real_debug_file << std::fixed << std::setprecision(6);
        // Write header
        sim2real_debug_file << "iteration\ttime\t"
                           << "value\t"
                           << "mse_obs_decoder\tmax_diff_obs_decoder\t"
                           << "obs_mean\tobs_std\tobs_min\tobs_max\t"
                           << "decoder_mean\tdecoder_std\tdecoder_min\tdecoder_max\t"
                           << "action_mean\taction_std\taction_min\taction_max\t"
                           << "latent_mean\tlatent_std\tlatent_min\tlatent_max\t"
                           << "base_lin_vel_x\tbase_lin_vel_y\tbase_lin_vel_z\t"
                           << "base_ang_vel_x\tbase_ang_vel_y\tbase_ang_vel_z\t"
                           << "non_zero_beyond_47"
                           << std::endl;
        file_opened = true;
    }
    
    if (debug_counter % 100 == 0) {  // Print every 100 iterations to avoid spam
        std::cout << "\n========== SIM2REAL GAP ANALYSIS (iteration " << debug_counter << ") ==========" << std::endl;
        
        // Compare first 47 elements (the actual observation part)
        float mse_first_47 = 0.0f;
        float max_diff_first_47 = 0.0f;
        for (size_t i = 0; i < num_cur_state; i++) {
            float diff = normalized_state_cur_[i] - normalized_critic_state_cur_[i];
            mse_first_47 += diff * diff;
            max_diff_first_47 = std::max(max_diff_first_47, std::abs(diff));
        }
        mse_first_47 /= num_cur_state;
        
        // Statistics for normalized observation (first 47 dims)
        float obs_sum = 0.0f, obs_sum_sq = 0.0f, obs_min = normalized_state_cur_[0], obs_max = normalized_state_cur_[0];
        for (size_t i = 0; i < num_cur_state; i++) {
            float val = normalized_state_cur_[i];
            obs_sum += val;
            obs_sum_sq += val * val;
            obs_min = std::min(obs_min, val);
            obs_max = std::max(obs_max, val);
        }
        float obs_mean = obs_sum / num_cur_state;
        float obs_std = std::sqrt((obs_sum_sq / num_cur_state) - (obs_mean * obs_mean));
        
        // Statistics for decoder output (first 47 dims)
        float decoder_sum = 0.0f, decoder_sum_sq = 0.0f, decoder_min = normalized_critic_state_cur_[0], decoder_max = normalized_critic_state_cur_[0];
        for (size_t i = 0; i < num_cur_state; i++) {
            float val = normalized_critic_state_cur_[i];
            decoder_sum += val;
            decoder_sum_sq += val * val;
            decoder_min = std::min(decoder_min, val);
            decoder_max = std::max(decoder_max, val);
        }
        float decoder_mean = decoder_sum / num_cur_state;
        float decoder_std = std::sqrt((decoder_sum_sq / num_cur_state) - (decoder_mean * decoder_mean));
        
        // Action statistics
        float action_sum = 0.0f, action_sum_sq = 0.0f, action_min = rl_action_(0), action_max = rl_action_(0);
        for (int i = 0; i < num_actuator_action; i++) {
            float val = rl_action_(i);
            action_sum += val;
            action_sum_sq += val * val;
            action_min = std::min(action_min, val);
            action_max = std::max(action_max, val);
        }
        float action_mean = action_sum / num_actuator_action;
        float action_std = std::sqrt((action_sum_sq / num_actuator_action) - (action_mean * action_mean));
        
        // Latent statistics
        float latent_sum = 0.0f, latent_sum_sq = 0.0f, latent_min = latent_cur_[0], latent_max = latent_cur_[0];
        for (size_t i = 0; i < num_cur_latent; i++) {
            float val = latent_cur_[i];
            latent_sum += val;
            latent_sum_sq += val * val;
            latent_min = std::min(latent_min, val);
            latent_max = std::max(latent_max, val);
        }
        float latent_mean = latent_sum / num_cur_latent;
        float latent_std = std::sqrt((latent_sum_sq / num_cur_latent) - (latent_mean * latent_mean));
        
        // Check if decoder output beyond index 47 is non-zero
        int non_zero_beyond_47 = 0;
        for (size_t i = num_cur_state; i < num_cur_critic_state; i++) {
            if (std::abs(normalized_critic_state_cur_[i]) > 1e-6) {
                non_zero_beyond_47++;
            }
        }
        
        // Print to console
        std::cout << "Observation (normalized, first 47):" << std::endl;
        std::cout << "  Mean: " << obs_mean << ", Std: " << obs_std 
                  << ", Min: " << obs_min << ", Max: " << obs_max << std::endl;
        std::cout << "Decoder output (first 47):" << std::endl;
        std::cout << "  Mean: " << decoder_mean << ", Std: " << decoder_std 
                  << ", Min: " << decoder_min << ", Max: " << decoder_max << std::endl;
        std::cout << "Reconstruction error:" << std::endl;
        std::cout << "  MSE: " << mse_first_47 << ", Max diff: " << max_diff_first_47 << std::endl;
        std::cout << "Actions:" << std::endl;
        std::cout << "  Mean: " << action_mean << ", Std: " << action_std 
                  << ", Min: " << action_min << ", Max: " << action_max << std::endl;
        std::cout << "Latent:" << std::endl;
        std::cout << "  Mean: " << latent_mean << ", Std: " << latent_std 
                  << ", Min: " << latent_min << ", Max: " << latent_max << std::endl;
        std::cout << "Value: " << value_ << std::endl;
        std::cout << "Non-zero decoder elements beyond 47: " << non_zero_beyond_47 << " / " << (num_cur_critic_state - num_cur_state) << std::endl;
        std::cout << "==========================================\n" << std::endl;
        
        // Write to file for offline analysis
        double current_time = rd_cc_.control_time_us_ / 1e6;
        sim2real_debug_file << debug_counter << "\t" << current_time << "\t"
                           << value_ << "\t"
                           << mse_first_47 << "\t" << max_diff_first_47 << "\t"
                           << obs_mean << "\t" << obs_std << "\t" << obs_min << "\t" << obs_max << "\t"
                           << decoder_mean << "\t" << decoder_std << "\t" << decoder_min << "\t" << decoder_max << "\t"
                           << action_mean << "\t" << action_std << "\t" << action_min << "\t" << action_max << "\t"
                           << latent_mean << "\t" << latent_std << "\t" << latent_min << "\t" << latent_max << "\t"
                           << base_lin_vel(0) << "\t" << base_lin_vel(1) << "\t" << base_lin_vel(2) << "\t"
                           << base_ang_vel(0) << "\t" << base_ang_vel(1) << "\t" << base_ang_vel(2) << "\t"
                           << non_zero_beyond_47
                           << std::endl;
    }
    debug_counter++;

    std::copy(normalized_critic_state_cur_.begin(),
                normalized_critic_state_cur_.begin() + num_cur_critic_state,
                input_states_buffer_c[0].begin());
    std::copy(normalized_critic_state_cur_.begin(),
                normalized_critic_state_cur_.begin() + num_cur_critic_state,
                input_states_buffer_dn[0].begin());
    // output tensor to value_
    output_tensors_c = session_c.Run(Ort::RunOptions{nullptr}, input_names_char_c.data(), input_tensors_c.data(), input_number_c, output_names_char_c.data(), output_number_c);
    output_tensors_dn = session_dn.Run(Ort::RunOptions{nullptr}, input_names_char_dn.data(), input_tensors_dn.data(), input_number_dn, output_names_char_dn.data(), output_number_dn);
    value_ = output_tensors_c[0].GetTensorMutableData<float>()[0];
    for (size_t i = 0; i < num_cur_critic_state; i++) {
        critic_state_cur_[i] = output_tensors_dn[0].GetTensorMutableData<float>()[i];
    }

    if (debug_counter % 1 == 0) {  // Print every 100 iterations to avoid spam
        std::cout << "Value from Critic: " << value_ << std::endl;
    }
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

            // Torque lower body 12 joints (indices 0-11)
            writeFile << rd_cc_.torque_desired.segment(0, 12).transpose() << "\t";
            
            // Joint position lower body 12 joints (indices 0-11)
            writeFile << q_noise_.segment(0, 12).transpose() << "\t";
            
            // Joint velocity lower body 12 joints (indices 0-11)
            writeFile << q_dot_lpf_.segment(0, 12).transpose() << "\t";
            
            // Base linear and angular velocity
            writeFile << base_lin_vel.transpose() << "\t" << base_ang_vel.transpose() << "\t";
            
            // Target commands
            writeFile << commands_(0) << "\t" << commands_(1) << "\t" << commands_(2) << "\t";
            
            // Right foot global position (x, y, z)
            writeFile << rd_cc_.link_[Right_Foot].xpos(0) << "\t" 
                      << rd_cc_.link_[Right_Foot].xpos(1) << "\t" 
                      << rd_cc_.link_[Right_Foot].xpos(2) << "\t";
            
            // Left foot global position (x, y, z)
            writeFile << rd_cc_.link_[Left_Foot].xpos(0) << "\t" 
                      << rd_cc_.link_[Left_Foot].xpos(1) << "\t" 
                      << rd_cc_.link_[Left_Foot].xpos(2) << "\t";
            // Base height z
            writeFile << rd_cc_.link_[Pelvis].xpos(2) << "\t";
            writeFile << std::endl;
            time_write_pre_ = rd_cc_.control_time_us_;
        }
    
    // Publish target velocities for MuJoCo visualization
    // Format: [vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
    // commands_ contains [target_vel_x, target_vel_y, target_vel_yaw]
    std_msgs::Float32MultiArray target_vel_msg;
    target_vel_msg.data.resize(6);
    target_vel_msg.data[0] = commands_(0);  // target_vel_x
    target_vel_msg.data[1] = commands_(1);  // target_vel_y
    target_vel_msg.data[2] = 0.0f;          // vel_z (not used for biped)
    target_vel_msg.data[3] = 0.0f;          // ang_vel_x (not used)
    target_vel_msg.data[4] = 0.0f;          // ang_vel_y (not used)
    target_vel_msg.data[5] = commands_(2);  // target_vel_yaw
    target_vel_pub_.publish(target_vel_msg);
    
    time_inference_pre_ = rd_cc_.control_time_us_;
    

}

void CustomController::computeSlow()

{

    copyRobotData(rd_);

    if (rd_cc_.tc_.mode == 7)

    {   
        static int count_walking_tick = 0;
        
        // cout << endl << endl;
        // cout << "Custom Controller Compute Slow CC Mode 7 --> " << "walking tick " << count_walking_tick << endl;
        // // print time
        // cout << (rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 << endl;
        // count_walking_tick++;
        // if (count_walking_tick > 40)
        //     exit(0);
        // double decay_factor = 0.999995; // Adjust this: closer to 1.0 is slower
        // double desired_hz_ = 125.0;
        // static int count_down_ticks = 0; // Number of ticks to reach desired_hz_

        // if (hz_ > desired_hz_) {
        //     // Multiply instead of subtract
        //     hz_ *= decay_factor;
        //     count_down_ticks++;

        //     // Safety floor
        //     if (hz_ < desired_hz_) hz_ = desired_hz_;

        //     del_t = 1.0 / hz_;

        //     if (count_down_ticks % 100 == 0) { // Don't spam the "Wall of Print" too fast
        //         std::cout << "Smoothly Scaling Hz: " << hz_ << " | dt: " << del_t << std::endl;
        //     }
        // }

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

            // use torque init to initialize rl action
            // torque_rl_(i) = DyrosMath::minmax_cut(rl_action_(i), -1., 1.) *torque_bound_(i) ;
            // for (int i = 0; i < num_actuator_action; i++)
            //     rl_action_(i) = torque_init_(i) / torque_bound_(i);
            cout << "Initial RL action from torque init: " << rl_action_.transpose() << endl;

            processNoise();

            processBias();

            processObservation();
        }

        processNoise();
        // if ((rd_cc_.control_time_us_/ 1.0e6 - time_pre_) >= (1/500.)-(1/pd_hz_)/2){
            
        //     cout <<"Noise processing done." << endl;
        // }
        
        // cout << "Time since last noise processing: " << (rd_cc_.control_time_us_/ 1.0e6 - time_pre_) << " seconds." << endl;

        processBias();
        // cout << "Time since last inference: " << (rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 << " seconds." << endl;
        if (use_margin_inference_){
            do_inference_ = (rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= (1/hz_)-(1/pd_hz_)/2;
            // cout << "Using margin for inference timing: " << (1/hz_)-(1/pd_hz_)/2 << " seconds." << endl;
            // cout << do_inference_ << endl;
        } else {
            do_inference_ = (rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= (1/hz_);
        }

        // if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= (1/hz_)-(1/pd_hz_)/2)

        // if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= (1/hz_))
        
        // cout << "[DEBUG] Base ang vel: " << (rd_cc_.q_dot_virtual_.segment(3,3)).transpose() << endl;
        if (do_inference_)
        {
            // cout << "Running inference" << endl;
            processObservation();

            if (!is_on_robot_){
                evaluateSimPerformance();
            }

            feedforwardPolicy();
            new_policy_updated_ = true;
            
            updateNextStepTime();

            // action_dt_accumulate_ += DyrosMath::minmax_cut(rl_action_(num_action-1)*5/hz_, 0.0, 5/hz_);

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

            // if (value_ < 2.6 && value_ > 0.){
            //     cout << "Low Value Warning: " << value_ << endl;
            //     exit(0);
            // }

        }

        for (int i = 0; i < num_actuator_action; i++){
            if (ctrl_type == 'T'){
                torque_rl_(i) = DyrosMath::minmax_cut(rl_action_(i), -1., 1.) *torque_bound_(i);
            }
            if (ctrl_type == 'P'){
                float q_std = (pd_limit(i, 1) - pd_limit(i, 0)) / 2;
                float q_bias = (pd_limit(i, 1) + pd_limit(i, 0)) / 2;
                torque_rl_(i) = DyrosMath::minmax_cut(kp_(i,i) * (DyrosMath::minmax_cut(rl_action_(i), -1., 1.) * q_std + q_bias - q_noise_(i)) - kv_(i,i)*q_vel_noise_(i), -torque_bound_(i), torque_bound_(i));
            }
        }

        // Low-pass filter the velocity commands to keep upper-body motion smooth
        const Eigen::Vector3d filtered_prev = command_vel_filtered_prev_;
        command_vel_filtered_ = command_filter_alpha_ * commands_ + (1.0 - command_filter_alpha_) * command_vel_filtered_;
        const Eigen::Vector3d filtered_delta = command_vel_filtered_ - filtered_prev;
        command_vel_filtered_prev_ = command_vel_filtered_;

        const double vel_scale_x = std::max(0.1, static_cast<double>(vel_scale_x_));
        const double vel_scale_y = std::max(0.05, static_cast<double>(vel_scale_y_));
        const double yaw_scale = std::max(0.1, static_cast<double>(max_stride_yaw));
        const double forward_norm = DyrosMath::minmax_cut(command_vel_filtered_(0) / vel_scale_x, -1.0, 1.0);
        const double lateral_norm = DyrosMath::minmax_cut(command_vel_filtered_(1) / vel_scale_y, -1.0, 1.0);
        const double yaw_norm = DyrosMath::minmax_cut(command_vel_filtered_(2) / yaw_scale, -1.0, 1.0);

        const double command_mag = command_vel_filtered_.norm();
        const double command_rate = filtered_delta.norm() * hz_;
        const double motion_mag_threshold = 0.18;
        const double motion_level = DyrosMath::minmax_cut(command_mag / motion_mag_threshold, 0.0, 1.0);
        const double change_level = DyrosMath::minmax_cut(command_rate / command_change_threshold_, 0.0, 1.0);
        const double engage = DyrosMath::minmax_cut(0.7 * motion_level + 0.3 * change_level, 0.0, 1.0);

        const double dt = 1.0 / hz_;
        const double swing_freq = 0.6 + 1.4 * std::abs(forward_norm);
        arm_swing_phase_ += TWO_PI * swing_freq * dt;
        if (arm_swing_phase_ > TWO_PI)
        {
            arm_swing_phase_ = std::fmod(arm_swing_phase_, TWO_PI);
        }

        double gait_phase = 0.0;
        double gait_wave_quadrature = 0.0;
        if (step_period_ > 1e-6)
        {
            gait_phase = (step_ticks_ + phase_indicator_ * step_period_) / (2.0 * step_period_);
            gait_phase = std::max(0.0, std::min(1.0, gait_phase));
            const double gait_angle = TWO_PI * gait_phase;
            gait_wave_quadrature = std::sin(gait_angle + HALF_PI);
        }

        Eigen::VectorQd q_upper_target = q_init_;
        applyUpperBodyMotion(q_upper_target,
                     engage,
                     forward_norm,
                     lateral_norm,
                     yaw_norm,
                     gait_phase,
                     gait_wave_quadrature);

        const Eigen::VectorVQd qdot_virtual = rd_cc_.q_dot_virtual_;
        const double cam_yaw = rd_cc_.CMM.row(5).dot(qdot_virtual);
        const double forward_cmd = command_vel_filtered_(0);
        const double control_time_us = rd_cc_.control_time_us_;
        const double cam_activation_threshold = 0.05;
        const double cam_release_threshold = 3.0;

        const bool forward_motion = std::abs(forward_cmd) > cam_activation_threshold;
        // Engage yaw CAM damping only when commanded to move forward/backward
        if (forward_motion)
        {
            cam_control_active_ = true;
            cam_quiet_timer_us_ = control_time_us;
        }

        // Project yaw centroidal momentum error into upper-body joints
        Eigen::VectorQd cam_gradient = rd_cc_.CMM.row(5).segment(6, MODEL_DOF).transpose();
        cam_gradient.head(num_actuator_action).setZero();

        Eigen::VectorQd cam_torque = Eigen::VectorQd::Zero();
        if (cam_control_active_)
        {
            const double gradient_norm = cam_gradient.segment(num_actuator_action, MODEL_DOF - num_actuator_action).squaredNorm();
            if (gradient_norm > 1e-6)
            {
                const double cam_gain = 120.0;
                cam_torque = (-cam_gain * cam_yaw / gradient_norm) * cam_gradient;
            }

            if (std::abs(cam_yaw) < cam_release_threshold)
            {
                if ((control_time_us - cam_quiet_timer_us_) > cam_release_duration_us_)
                {
                    cam_control_active_ = false;
                }
            }
            else
            {
                cam_quiet_timer_us_ = control_time_us;
            }
        }

        const double posture_scale = cam_control_active_ ? 0.4 : 1.0;
        const double damping_scale = cam_control_active_ ? 0.4 : 0.8;
        // if (upper_body_motion_){

        //     for (int i = num_actuator_action; i < MODEL_DOF; i++)
        //     {
        //         const double posture = posture_scale * kp_(i, i) * (q_upper_target(i) - q_noise_(i));
        //         const double damping = damping_scale * (-kv_(i, i) * q_vel_noise_(i));
        //         // const double torque = posture + damping + cam_torque(i);
        //         const double torque = posture + damping;
        //         torque_rl_(i) = DyrosMath::minmax_cut(torque, -torque_bound_(i), torque_bound_(i));
        //     }
        // }
        // else{
            // }


        
        for (int i = num_actuator_action; i < MODEL_DOF; i++)
        {
            torque_rl_(i) = kp_(i, i) * (q_init_(i) - q_noise_(i)) - kv_(i, i) * q_vel_noise_(i);
        }
        
        if (rd_cc_.control_time_us_ < start_time_ + 0.0e6)
        {
            for (int i = 0; i <MODEL_DOF; i++)
                torque_spline_(i) = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 0.3e6, torque_init_(i), torque_rl_(i), 0.0, 0.0);

            rd_.torque_desired = torque_spline_;   
            torque_sum_lpf_ = torque_spline_.head(12); 
        }
        else{
                    //LPF
        // double torque_cutoff_freq = 40.0;
        // torque_sum_lpf_ = 1 / (1 + 2 * M_PI * torque_cutoff_freq * del_t) * torque_sum_lpf_ //previous tick torque
        //                 + (2 * M_PI * torque_cutoff_freq * del_t) / (1 + 2 * M_PI * torque_cutoff_freq * del_t) * torque_sum_; //updated torque
            if (use_lpf_){
                // cout << "use torque lpf" << endl;
                
                // //slowly increase cutoff frequency to 110 Hz over 10 seconds
                // if ((rd_cc_.control_time_us_ - start_time_) < 10.0e6){
                //     torque_cutoff_freq = 10.0 + 10.0 * ((rd_cc_.control_time_us_ - start_time_) / 1.0e6);
                // }else{
                //     torque_cutoff_freq = 110.0;
                // }
                // cout << "Torque LPF cutoff frequency: " << torque_cutoff_freq << " Hz" << endl;
                for (int i = 0; i < 12; i++){
                    torque_sum_lpf_(i) = 1 / (1 + 2 * M_PI * torque_cutoff_freq * (1/2000.0)) * torque_sum_lpf_(i) //previous tick torque
                                    + (2 * M_PI * torque_cutoff_freq * (1/2000.0)) / (1 + 2 * M_PI * torque_cutoff_freq * (1/2000.0)) * torque_rl_(i); //updated torque
                }
                // rd_.torque_desired = torque_sum_lpf_;
                rd_.torque_desired = torque_rl_;
                // cout << "before lpf: " << torque_rl_.head(12).transpose() << endl;
                rd_.torque_desired.head(12) = torque_sum_lpf_;
                // cout << "after lpf: " << rd_.torque_desired.head(12).transpose() << endl;
                
            }
            else{
                if (use_inter_){
                    // cout << "use interpolation" << endl;
                    // compute interpolation based on the difference between hz_ and pd_hz_
                    static int ticks_since_last_policy = 0;
                    int ticks_per_update = (int)(pd_hz_ / hz_);
                    if (new_policy_updated_) { // Boolean you set when the RL network outputs a new vector
                        torque_rl_prev_ = rd_.torque_desired.head(12); // Store the previous torque command
                        ticks_since_last_policy = 1;
                        new_policy_updated_ = false;
                    }
                    // 3. Linear Interpolation formula: (1 - alpha) * Old + (alpha) * New
                    double alpha = (double)ticks_since_last_policy / (double)ticks_per_update;
                    if (alpha > 1.0) alpha = 1.0; // Safety cap

                    for (int i = 0; i < 12; i++) {
                        double interpolated_torque = (1.0 - alpha) * torque_rl_prev_(i) + alpha * torque_rl_(i);
                        rd_.torque_desired(i) = interpolated_torque;
                    }

                    // Keep the rest of the body at the current raw command
                    for (int i = 12; i < MODEL_DOF; i++) {
                        rd_.torque_desired(i) = torque_rl_(i);
                    }

                    
                    // cout << "Ticks since last policy: " << ticks_since_last_policy << "/" << ticks_per_update << ", alpha: " << alpha << endl;
                    ticks_since_last_policy++;
                    
                    // rd_.torque_desired = torque_rl_;
                } else{
                    if (use_max_AR_){
                        if (new_policy_updated_) { // Boolean you set when the RL network outputs a new vector
                            torque_rl_prev_ = rd_.torque_desired.head(12);
                            
                        }
                        rd_.torque_desired = torque_rl_;
                        for (int i = 0; i < 12; i++) {
                            double diff = (torque_rl_(i) - torque_rl_prev_(i)) * hz_;
                            if (std::abs(diff) > max_AR_) {
                                if (diff > 0) {
                                    rd_.torque_desired(i) = torque_rl_prev_(i) + max_AR_ / hz_;
                                } else {
                                    rd_.torque_desired(i) = torque_rl_prev_(i) - max_AR_ / hz_;
                                }
                            }
                        }   
                        
                        if (new_policy_updated_ ){
                            new_policy_updated_ = false;
                            // udate rl_action_ to keep input to the network consistent
                            // for (int i = 0; i < num_actuator_action; i++){
                            //     rl_action_(i) = rd_.torque_desired(i) / torque_bound_(i);
                            // }
                        }
                        
                        // rd_.torque_desired.head(12) = torque_max_AR_;


                    } else{
                        // cout << "no lpf and no interpolation" << endl;
                        rd_.torque_desired = torque_rl_;
                    }

                }
            }
        }

        if (stop_by_value_thres_)
            rd_.torque_desired = kp_ * (q_stop_ - q_noise_) - kv_*q_vel_noise_;

        // if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= 1/hz_)
        
        if (do_inference_)
            processEverythingElse();


        plotTorques();

    }

}

void CustomController::applyUpperBodyMotion(Eigen::VectorQd &q_target,
                                            double engage,
                                            double forward_norm,
                                            double lateral_norm,
                                            double yaw_norm,
                                            double gait_phase,
                                            double gait_wave_quadrature)
{
    const double lean_gain = 0.0;
    const double lean_offset = engage * DyrosMath::minmax_cut(-lean_gain * forward_norm, -0.1, 0.1);
    q_target(IDX_WAIST2) += lean_offset;

    const double waist_yaw_gain = 0.2;
    const double waist_yaw_offset = engage * DyrosMath::minmax_cut(waist_yaw_gain * yaw_norm, -0.3, 0.3);
    q_target(IDX_WAIST1) += waist_yaw_offset;

    const double torso_roll_gain = 0.05;
    const double torso_roll_offset = engage * DyrosMath::minmax_cut(torso_roll_gain * lateral_norm, -0.18, 0.18);
    q_target(IDX_UPPERBODY) += torso_roll_offset;

    const double arm_swing_gain = 0.0;
    const double arm_swing_amp = engage * DyrosMath::minmax_cut(arm_swing_gain * forward_norm, -0.5, 0.5);
    const double swing_wave = arm_swing_amp * std::sin(arm_swing_phase_);
    q_target(IDX_L_SHOULDER1) += swing_wave;
    q_target(IDX_R_SHOULDER1) -= swing_wave;

    const double arm_pitch_gain = -0.;
    const double arm_pitch_offset = engage * DyrosMath::minmax_cut(arm_pitch_gain * forward_norm, -0.3, 0.3);
    const double shake_intensity = engage * (0.08 + 0.22 * std::abs(forward_norm));
    const double command_dir = (forward_norm >= 0.0) ? 1.0 : -1.0;
    const double shoulder_signal = std::sin(TWO_PI * gait_phase);
    const double shoulder_phys_left = shake_intensity * command_dir * shoulder_signal;
    const double shoulder_phys_right = -shoulder_phys_left;
    const double shoulder_shake_left = DyrosMath::minmax_cut(shoulder_phys_left / AXIS_SIGN_L_SHOULDER2, -0.35, 0.35);
    const double shoulder_shake_right = DyrosMath::minmax_cut(shoulder_phys_right / AXIS_SIGN_R_SHOULDER2, -0.35, 0.35);
    q_target(IDX_L_SHOULDER2) += arm_pitch_offset - shoulder_shake_left;
    q_target(IDX_R_SHOULDER2) += arm_pitch_offset - shoulder_shake_right;

    const double elbow_gain = -0.;
    const double elbow_bias = engage * DyrosMath::minmax_cut(elbow_gain * forward_norm, -0.18, 0.18);
    const double elbow_wave = engage * 0.32 * std::abs(forward_norm) * std::sin(arm_swing_phase_ + HALF_PI);
    const double elbow_phys_left = shake_intensity * 0.6 * gait_wave_quadrature * command_dir;
    const double elbow_phys_right = -elbow_phys_left;
    const double elbow_shake_left = DyrosMath::minmax_cut(elbow_phys_left / AXIS_SIGN_L_ELBOW, -0.2, 0.2);
    const double elbow_shake_right = DyrosMath::minmax_cut(elbow_phys_right / AXIS_SIGN_R_ELBOW, -0.2, 0.2);
    q_target(IDX_L_ELBOW) += elbow_bias + elbow_wave + elbow_shake_left;
    q_target(IDX_R_ELBOW) += -elbow_bias - elbow_wave + elbow_shake_right;
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

void CustomController::updateCommandFromTimeline(const std::string &command_file)
{
    static std::vector<Eigen::Vector3d> timeline;
    static bool loaded = false;
    static double start_walking_time = -1.0; // Captures the start of the test

    // 1. Load the file (Fixed to handle comments/formatting)
    if (!loaded) {
        std::ifstream file(command_file);
        if (!file) {
            ROS_ERROR("Failed to open timeline file: %s", command_file.c_str());
            loaded = true; return;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            // Remove comments and skip empty lines
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;

            std::stringstream ss(line);
            double vx, vy, vyaw;
            if (ss >> vx >> vy >> vyaw) {
                timeline.push_back(Eigen::Vector3d(vx, vy, vyaw));
            }
        }
        file.close();
        loaded = true;
        ROS_INFO("Timeline loaded: %lu steps detected.", timeline.size());
    }

    // 2. Synchronize time
    double current_time = rd_cc_.control_time_us_ / 1e6;

    // Only start the clock if we are actually in walking mode
    // (Replace 'is_walking_state' with your actual variable name, e.g., walking_tick > 0)
    if (start_walking_time < 0) {
        start_walking_time = current_time;
    }

    double elapsed = current_time - start_walking_time;
    
    // 3. Select command (every 5 seconds relative to start)
    int index = static_cast<int>(elapsed / 5.0);

    if (index < timeline.size()) {
        commands_(0) = timeline[index](0);
        commands_(1) = timeline[index](1);
        commands_(2) = timeline[index](2);
    } else {
        // Optional: Stop the robot when timeline ends
        commands_.setZero();
    }
}

void CustomController::updateNextStepTime()
{           
    step_ticks_ += del_t;
    if (step_ticks_ >= step_period_) {
        step_ticks_ = 0.;
        phase_indicator_ = 1-phase_indicator_;
        cout << endl << endl << "__________________________________" << endl;
        cout << "Step phase switched to: " << phase_indicator_ << endl << endl;
    }
}

// void CustomController::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
// {
//     cout << "Joy Callback!" << endl;
//     commands_(0) = DyrosMath::minmax_cut(vel_scale_x_*joy->axes[1], -0.5, 0.5);
//     commands_(1) = DyrosMath::minmax_cut(vel_scale_y_*joy->axes[0] , -0.5, 0.5);

//     if (joy->buttons[1] == 1.0 && vel_scale_x_ < 1.0 && vel_scale_y_ < 0.3){
//         vel_scale_x_ += 0.03;
//         vel_scale_y_ += 0.01;
//         ROS_INFO("Velocity X : %f", vel_scale_x_);
//         ROS_INFO("Velocity Y : %f", vel_scale_y_);
//     }

//     if (joy->buttons[0] == 1.0 && vel_scale_x_ > 0.1 && vel_scale_y_ > 0.03){
//         vel_scale_x_ -= 0.03;
//         vel_scale_y_ -= 0.01;
//         ROS_INFO("Velocity X : %f", vel_scale_x_);
//         ROS_INFO("Velocity Y : %f", vel_scale_y_);
//     }
//     if(joy->buttons[6] == 1){
//         commands_(2) = 0.6;
//     }
//     if(joy->buttons[7] == 1){
//         commands_(2) = -0.6;
//     }
//     if(joy->buttons[6] != 1 && joy->buttons[7] != 1){
//         commands_(2) = 0.;
//     }
// }

void CustomController::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
    cout << "Joy Callback!" << endl;
    commands_(0) = DyrosMath::minmax_cut(vel_scale_x_*joy->axes[1], -0.5, 1.);
    commands_(1) = DyrosMath::minmax_cut(vel_scale_y_*joy->axes[0] , -0.5, 0.5);

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
    // if(joy->buttons[6] == 1){
    //     commands_(2) = 0.6;
    // }
    // if(joy->buttons[7] == 1){
    //     commands_(2) = -0.6;
    // }
    // if(joy->buttons[6] != 1 && joy->buttons[7] != 1){
    //     commands_(2) = 0.;
    // }
    if (joy->axes[2] < 1. && joy->axes[2] != 0.){
        commands_(2) = DyrosMath::minmax_cut(-(joy->axes[2]-1), 0., 2.)/4.;
    } else if (joy->axes[5] < 1. && joy->axes[5] != 0.){
        commands_(2) = -DyrosMath::minmax_cut(-(joy->axes[5]-1), 0., 2.)/4.;
    } else {
        commands_(2) = 0.;
    }
}


Eigen::VectorQd CustomController::getControl()
{
    return ControlVal_;
}

void CustomController::plotTorques()
{
    static std::vector<Eigen::VectorXd> torque_history;
    static std::vector<Eigen::VectorXd> torque_raw_history;
    // plot the h_cur_ values too (first 5 values only)
    static std::vector<Eigen::VectorXd> hcur_history; 
    static std::vector<Eigen::VectorXd> q_dot_history_;
    static std::vector<Eigen::VectorXd> q_dot_history_virtual_;
    static std::vector<Eigen::VectorXd> q_dot_lpf_history_;
    static double plot_start_time = -1.0;
    static bool plot_finished = false;

    if (plot_finished) return;

    double current_time = rd_cc_.control_time_us_ / 1e6;
    if (plot_start_time < 0) plot_start_time = current_time;

    double elapsed = current_time - plot_start_time;
    // cout << "Elapsed time for torque logging: " << elapsed << " seconds" << std::endl;
    if (elapsed <= 10.) 
    {
        torque_history.push_back(rd_cc_.torque_desired.head(12));
        if (use_lpf_){
            torque_raw_history.push_back(torque_rl_.head(12));
        }
        // This creates a temporary vector containing the first 5 elements of h_cur_
        hcur_history.push_back(Eigen::Map<Eigen::VectorXf>(h_cur_.data(), 5).cast<double>());

        //
        // print shape of q_vel_noise_, q_dot_virtual_, q_dot_lpf_
        // cout << "q_vel_noise_ size: " << q_vel_noise_.size() << std::endl;
        // cout << "q_dot_virtual_ size: " << rd_cc_.q_dot_virtual_.size() << std::endl;
        // cout << "q_dot_lpf_ size: " << q_dot_lpf_.size() << std::endl;
        // exit(0);
        q_dot_history_.push_back(q_vel_noise_);
        // for virtual, ignore first 6 values (base)
        q_dot_history_virtual_.push_back(rd_cc_.q_dot_virtual_.segment(6,18));
        q_dot_lpf_history_.push_back(q_dot_lpf_);

    }
    else 
    {
        // 1. Open the file
        std::ofstream outFile("/home/dyros/catkin_ws/src/tocabi_cc/visu/torque_log.csv");

        if (outFile.is_open()) {
            // 2. Write the Header
            outFile << "Tick";
            for(int j=0; j<12; j++) outFile << ",Joint_" << j;
            if (use_lpf_){
                for(int j=0; j<12; j++) outFile << ",Joint_" << j << "_raw";
            }
            for(int j=0; j<12; j++) outFile << ",Qdot_Joint_" << j;
            for(int j=0; j<12; j++) outFile << ",Qdot_LPF_Joint_" << j;
            for(int j=0; j<12; j++) outFile << ",Qdot_Virtual_Joint_" << j;
            outFile << "\n";

            // 3. Write the Data
            for (size_t i = 0; i < torque_history.size(); ++i) {
                outFile << i;
                for (int j = 0; j < 12; ++j) {
                    outFile << "," << torque_history[i](j);
                }
                if (use_lpf_){
                    for (int j = 0; j < 12; ++j) {
                        outFile << "," << torque_raw_history[i](j);
                    }
                }
                for (int j = 0; j < 12; ++j) {
                    outFile << "," << q_dot_history_[i](j);
                }
                for (int j = 0; j < 12; ++j) {
                    outFile << "," << q_dot_lpf_history_[i](j);
                }
                for (int j = 0; j < 12; ++j) {
                    outFile << "," << q_dot_history_virtual_[i](j);
                }
                outFile << "\n";
            }

            // if(use_lpf_){
            //     // record torque_rl_ too
            //     outFile << "\nTick";;
            //     for(int j=0; j<12; j++) outFile << ",Joint_" << j << "_raw";;
            //     outFile << "\n";
            //     for (size_t i = 0; i < torque_raw_history.size(); ++i) {
            //         outFile << i;
            //         for (int j = 0; j < 12; ++j) {
            //             outFile << "," << torque_raw_history[i](j);
            //         }
            //         outFile << "\n";
            //     }
            // }
            outFile.close();
            std::cout << "\n[SUCCESS] Torque data saved to torque_log.csv" << std::endl;
        } else {
            std::cerr << "\n[ERROR] Could not open file for writing!" << std::endl;
        }

        torque_history.clear();
        torque_history.shrink_to_fit();

        std::ofstream outFile2("/home/dyros/catkin_ws/src/tocabi_cc/visu/h_log.csv");
        if (outFile2.is_open()) {
            // 2. Write the Header
            outFile2 << "Tick";
            for(int j=0; j<5; j++) outFile2 << ",Hcur_" << j;
            outFile2 << "\n";

            // 3. Write the Data
            for (size_t i = 0; i < hcur_history.size(); ++i) {
                outFile2 << i;
                for (int j = 0; j < 5; ++j) {
                    outFile2 << "," << hcur_history[i](j);
                }
                outFile2 << "\n";
            }

            outFile2.close();
            std::cout << "\n[SUCCESS] Hcur data saved to h_log.csv" << std::endl;
        } else {
            std::cerr << "\n[ERROR] Could not open file for writing!" << std::endl;
        }

        plot_finished = true;
        // exit(0); // Exit the program after saving the file
    }
}


void CustomController::evaluateSimPerformance()
{
    static double sum_err_x = 0, sum_err_y = 0, sum_err_yaw = 0;
    static int sample_count = 0;
    static double start_time = -1.0;
    static bool is_finished = false;

    static double peak_fz = 0.0;
    static int high_force_count = 0;
    static double sum_sq_fz = 0.0;
    static double sum_fz = 0.0;

    static Eigen::VectorXd prev_action = Eigen::VectorXd::Zero(num_actuator_action);
    static double peak_action_rate = 0.0;
    static int high_act_R_count = 0;
    static double sum_action_R = 0.0;
    static double sum_sq_action_R = 0.0;
    static double sum_action = 0.0;
    static double sum_sq_action = 0.0;

    static Eigen::VectorXd peak_action_R_per_joint = Eigen::VectorXd::Zero(num_actuator_action);
    static Eigen::VectorXd sum_action_R_per_joint = Eigen::VectorXd::Zero(num_actuator_action);
    static Eigen::VectorXd sum_sq_action_R_per_joint = Eigen::VectorXd::Zero(num_actuator_action);
    static Eigen::VectorXd sum_action_per_joint = Eigen::VectorXd::Zero(num_actuator_action);
    static Eigen::VectorXd sum_sq_action_per_joint = Eigen::VectorXd::Zero(num_actuator_action);

    static int tick_count = 0;
    tick_count++;


    if (is_finished) {
        std::cout  
            << "Mean X: " << sum_err_x / sample_count 
            << " | Mean Y: " << sum_err_y / sample_count 
            << " | Mean Yaw: " << sum_err_yaw / sample_count 
            << " | Peak Fz: " << (int)peak_fz 
            << " | >1300: " << high_force_count 
            << " | Mean Fz: " << (int)(sum_fz / sample_count)
            << " | Mean Sq Fz: " << (int)(sum_sq_fz / sample_count) 
            << " | Max action R: " << peak_action_rate 
            << " | >375: " << high_act_R_count
            << " | Mean Act R: " << sum_action_R / sample_count
            << " | Mean Sq Act R: " << (int)(sum_sq_action_R / sample_count)
            << " | Mean Act: " << sum_action / sample_count
            << " | Mean Sq Act: " << sum_sq_action / sample_count
            << "\r" << std::flush;

            std::cout << "\n--- PER-JOINT ACTION RATE REPORT ---" << std::endl;
            std::cout << "Joint | Max Action Rate | Mean Action Rate | Mean Sq Action Rate | Mean Action | Mean Sq Action" << std::endl;
            for (int i = 0; i < num_actuator_action; ++i) {
                std::cout << std::setw(5) << i << " | " 
                        << std::setw(15) << peak_action_R_per_joint(i) << " | " 
                        << std::setw(16) << sum_action_R_per_joint(i) / sample_count << " | "
                        << std::setw(19) << sum_sq_action_R_per_joint(i) / sample_count << " | "
                        << std::setw(11) << sum_action_per_joint(i) / sample_count << " | "
                        << std::setw(14) << sum_sq_action_per_joint(i) / sample_count
                        << std::endl;
            }
            cout << "\r" << std::flush;
        return;
    }

    double current_time = rd_cc_.control_time_us_ / 1e6;

    if (start_time < 0) start_time = current_time;

    double elapsed = current_time - start_time;

    if (elapsed <= 180.0)
    {
        double err_x = std::abs(commands_(0) - base_lin_vel(0));
        double err_y = std::abs(commands_(1) - base_lin_vel(1));
        double err_yaw = std::abs(commands_(2) - base_ang_vel(2));

        sum_err_x += err_x;
        sum_err_y += err_y;
        sum_err_yaw += err_yaw; 
        sample_count++;

        double current_fz = std::max(std::abs(rd_cc_.LF_CF_FT(2)), std::abs(rd_cc_.RF_CF_FT(2)));
        if (current_fz > peak_fz) peak_fz = current_fz;
        if (current_fz > 1300.0) high_force_count++;
        sum_fz += current_fz;
        sum_sq_fz += (current_fz * current_fz);

        // Calculate Torque Rate (N.m / s)
        // (Action_t - Action_t-1) * frequency = Rate of Change
        // cout << "torque rl : " << torque_rl_.transpose() << endl;
        // cout << "prev action : " << prev_action.transpose() << endl;
        Eigen::VectorXd torque_legs_rl_ = torque_rl_.head(12);
        // Eigen::VectorXd torque_legs_rl_ = rd_cc_.torque_desired.head(12);
        Eigen::VectorXd torque_rates = (torque_legs_rl_ - prev_action).cwiseAbs() * hz_;
        

        // Identify the most aggressive motor torque change
        double max_action_rate = torque_rates.lpNorm<Eigen::Infinity>();
        if (max_action_rate > peak_action_rate) peak_action_rate = max_action_rate;
        // Cumulative "Control Effort" or "Jitter" metric
        //L1 norm then L2
        sum_action_R += torque_rates.mean();
        sum_sq_action_R += torque_rates.squaredNorm();
        
        sum_action += torque_legs_rl_.cwiseAbs().mean();
        sum_sq_action += torque_legs_rl_.squaredNorm();

        sum_action_R_per_joint += torque_rates;
        sum_sq_action_R_per_joint += torque_rates.cwiseProduct(torque_rates);

        peak_action_R_per_joint = peak_action_R_per_joint.cwiseMax(torque_rates);

        sum_action_per_joint += torque_legs_rl_.cwiseAbs();
        sum_sq_action_per_joint += torque_legs_rl_.cwiseProduct(torque_legs_rl_);

        // Threshold: 375 N.m/s 
        if (max_action_rate > 375.0) high_act_R_count++;
        prev_action = torque_legs_rl_;
        // cout << "============ " << torque_legs_rl_.cwiseAbs().mean() << endl;
        // cout << "torque rates : " << torque_rates.transpose() << endl;
        // cout << "torque rl : " << torque_legs_rl_.transpose() << endl;
        // cout << "prev action : " << prev_action.transpose() << endl;

        if ((tick_count+1) % 100 == 0) {
            // cout << " all action diff : " << action_diff.transpose() << endl;
            // cout << " l_inf action diff : " << current_l_inf << endl;
            // cout << " max action diff : " << action_diff.maxCoeff() << endl;
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "[RECORDING] Time: " << elapsed 
                      << " walking tick: " << tick_count
                      << " | Mean X: " << sum_err_x / sample_count 
                      << " | Mean Y: " << sum_err_y / sample_count 
                      << " | Mean Yaw: " << sum_err_yaw / sample_count 
                      << " | Peak Fz: " << (int)peak_fz 
                      << " | >1300: " << high_force_count 
                      << " | Mean Fz: " << (int)(sum_fz / sample_count)
                      << " | Mean Sq Fz: " << (int)(sum_sq_fz / sample_count) 
                      << " | Max action R: " << peak_action_rate 
                      << " | >375: " << high_act_R_count
                      << " | Mean Act R: " << sum_action_R / sample_count
                      << " | Mean Sq Act R: " << (int)(sum_sq_action_R / sample_count)
                      << " | Mean Act: " << sum_action / sample_count
                      << " | Mean Sq Act: " << sum_sq_action / sample_count
                      << " |\r" << std::flush;

            std::cout << "\n--- PER-JOINT ACTION RATE REPORT ---" << std::endl;
            std::cout << "Joint | Max Action Rate | Mean Action Rate | Mean Sq Action Rate | Mean Action | Mean Sq Action" << std::endl;
            for (int i = 0; i < num_actuator_action; ++i) {
                std::cout << std::setw(5) << i << " | " 
                        << std::setw(15) << peak_action_R_per_joint(i) << " | " 
                        << std::setw(16) << sum_action_R_per_joint(i) / sample_count << " | "
                        << std::setw(19) << sum_sq_action_R_per_joint(i) / sample_count << " | "
                        << std::setw(11) << sum_action_per_joint(i) / sample_count << " | "
                        << std::setw(14) << sum_sq_action_per_joint(i) / sample_count << std::endl;
            }
            cout << "\r" << std::flush;
        }

    }
    else{
        is_finished = true; 
    }
}