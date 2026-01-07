#include "cc.h"

using namespace TOCABI;

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
            writeFile.open("/home/rui/ubuntu-20-04/raibertGRU_ws/src/tocabi_cc/result/rui_data.csv", std::ofstream::out);
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

    joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy_wh", 10, &CustomController::joyCallback, this);
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
    kv_.diagonal() << 15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        200.0, 100.0, 100.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
                        2.0, 2.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0;
    kv_.diagonal() /= 3.0;

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


}

void CustomController::loadOnnX()
{
    string cur_path = "/home/rui/ubuntu-20-04/raibertGRU_ws/src/tocabi_cc/";
    string actor_path = "/home/rui/rui_ws/erfi_ws/AMP_framework/logs/onnx/actor.onnx";
    string normalizer_path = "/home/rui/rui_ws/erfi_ws/AMP_framework/logs/onnx/normalizer.onnx";
    string denormalizer_path = "/home/rui/rui_ws/erfi_ws/AMP_framework/logs/onnx/denormalizer.onnx";
    string decoder_path = "/home/rui/rui_ws/erfi_ws/AMP_framework/logs/onnx/decoder.onnx";
    string critic_path = "/home/rui/rui_ws/erfi_ws/AMP_framework/logs/onnx/critic.onnx";


    if (is_on_robot_)
    {
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/";
        actor_path = cur_path + "onnx_files/actor.onnx"; 
        normalizer_path = cur_path + "onnx_files/normalizer.onnx";
        denormalizer_path = cur_path + "onnx_files/denormalizer.onnx";
        decoder_path = cur_path + "onnx_files/decoder.onnx";
        critic_path = cur_path + "onnx_files/critic.onnx";
    }

    if (ctrl_mode){
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
                
}

void CustomController::feedforwardPolicy()
{


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
            if (ctrl_type == 'T'){
                torque_rl_(i) = DyrosMath::minmax_cut(rl_action_(i), -1., 1.) *torque_bound_(i) ;
            }
            if (ctrl_type == 'P'){
                float q_std = (pd_limit(i, 1) - pd_limit(i, 0)) / 2;
                float q_bias = (pd_limit(i, 1) + pd_limit(i, 0)) / 2;
                torque_rl_(i) = DyrosMath::minmax_cut(kp_(i,i) * (DyrosMath::minmax_cut(rl_action_(i), -1., 1.) * q_std + q_bias - q_noise_(i)) - kv_(i,i)*q_vel_noise_(i), -torque_bound_(i), torque_bound_(i));
            }
            
        }
        
        for (int i = num_actuator_action; i < MODEL_DOF; i++)
            torque_rl_(i) = kp_(i,i) * (q_init_(i) - q_noise_(i)) - kv_(i,i)*q_vel_noise_(i);
        
        if (rd_cc_.control_time_us_ < start_time_ + 0.3e6)
        {
            for (int i = 0; i <MODEL_DOF; i++)
                torque_spline_(i) = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 0.3e6, torque_init_(i), torque_rl_(i), 0.0, 0.0);

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