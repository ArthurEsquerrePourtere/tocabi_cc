#include "tocabi_lib/robot_data.h"
#include "wholebody_functions.h"
#include <random>
#include <cmath>

#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/Float32MultiArray.h>
#include "onnxruntime_cxx_api.h"

class CustomController
{
public:
    CustomController(RobotData &rd);
    Eigen::VectorQd getControl();

    //void taskCommandToCC(TaskCommand tc_);

    double hz_ = 125.;
    const double pd_hz_ = 2000.;
    double del_t = 1 / hz_;

    void computeSlow();
    void evaluateSimPerformance();
    void plotTorques();
    void computeFast();
    void computePlanner();
    void copyRobotData(RobotData &rd_l);

    RobotData &rd_;
    RobotData rd_cc_;

    void loadOnnX();
    void processNoise();
    void processBias();
    void initBias();
    Eigen::Matrix<double, MODEL_DOF, 1> q_bias_;
    void processObservation();
    void initVariable();
    void feedforwardPolicy();
    void processEverythingElse();
    void updateNextStepTime();
    void applyUpperBodyMotion(Eigen::VectorQd &q_target,
                              double engage,
                              double forward_norm,
                              double lateral_norm,
                              double yaw_norm,
                              double gait_phase,
                              double gait_wave_quadrature);

    Eigen::Vector3d mat2euler(Eigen::Matrix3d mat);


    /////////////////////////////////// ONNX Runtime by Yongarry ///////////////////////////////////////
    size_t input_number, output_number;
    std::vector<std::string> input_names, output_names;
    std::vector<const char *> input_names_char, output_names_char;
    std::vector<Ort::Value> input_tensors, output_tensors;
    std::vector<std::vector<float>> input_states_buffer;

    size_t input_number_n, output_number_n;
    std::vector<std::string> input_names_n, output_names_n;
    std::vector<const char *> input_names_char_n, output_names_char_n;
    std::vector<Ort::Value> input_tensors_n, output_tensors_n;
    std::vector<std::vector<float>> input_states_buffer_n;

    size_t input_number_d, output_number_d;
    std::vector<std::string> input_names_d, output_names_d;
    std::vector<const char *> input_names_char_d, output_names_char_d;
    std::vector<Ort::Value> input_tensors_d, output_tensors_d;
    std::vector<std::vector<float>> input_states_buffer_d;

    size_t input_number_c, output_number_c;
    std::vector<std::string> input_names_c, output_names_c;
    std::vector<const char *> input_names_char_c, output_names_char_c;
    std::vector<Ort::Value> input_tensors_c, output_tensors_c;
    std::vector<std::vector<float>> input_states_buffer_c;

    size_t input_number_dn, output_number_dn;
    std::vector<std::string> input_names_dn, output_names_dn;
    std::vector<const char *> input_names_char_dn, output_names_char_dn;
    std::vector<Ort::Value> input_tensors_dn, output_tensors_dn;
    std::vector<std::vector<float>> input_states_buffer_dn;

    std::vector<float> state_cur_, critic_state_cur_, latent_cur_, h_cur_;
    std::vector<float> normalized_state_cur_, normalized_critic_state_cur_;

    int input_obs_idx_ = 0;
    int input_h0_idx_ = 1;
    int output_action_idx_ = 0;
    int output_hn_idx_ = 1;
    int output_latent_idx_ = 2;

    static const int num_action = 12;
    static const int num_actuator_action = 12;
    static const int num_cur_state = 47;
    static const int num_cur_critic_state = 166;
    static const int num_cur_latent = 24;
    static const int num_cur_h = 256;

    Eigen::MatrixXd rl_action_;
    Eigen::MatrixXd obs_rl_action_;
    double value_;

    bool stop_by_value_thres_ = false;
    Eigen::Matrix<double, MODEL_DOF, 1> q_stop_;
    float stop_start_time_;

    std::ofstream writeFile;
    std::ofstream evalFile;

    bool is_on_robot_ = false;
    bool is_write_file_ = false;
    Eigen::Matrix<double, MODEL_DOF, 1> q_dot_lpf_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_dot_lpf_higher_;

    Eigen::Matrix<double, MODEL_DOF, 1> q_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_pre_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_vel_noise_;
    Eigen::Vector12d q_leg_desired_;

    Eigen::Matrix<double, MODEL_DOF, 1> torque_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_spline_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_rl_;
    Eigen::Matrix<double, 12, 1> torque_sum_lpf_;
    Eigen::Matrix<double, 12, 1> torque_rl_prev_;
    Eigen::Matrix<double, 12, 1> torque_max_AR_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_bound_;
    Eigen::Matrix<double, num_action, 2> pd_limit;
    const char ctrl_type = 'T';

    bool use_lpf_ = true;
    bool use_inter_ = false;
    bool use_max_AR_ = false;
    bool use_margin_inference_ = false;
    bool do_inference_ = false;
    bool new_policy_updated_ = false;
    
    bool use_lpf_vel_obs_ = false;
    double ang_vel_lpf_cutoff_freq_ = 60.0;
    double leg_dof_vel_lpf_cutoff_freq_ = 15.0;
    double feet_dof_vel_lpf_cutoff_freq_ = 60.0;

    double torque_cutoff_freq = 80.0; //for torque lpf
    double max_AR_ = 1200;


    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kp_;
    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kv_;

    float start_time_;
    float time_inference_pre_ = 0.0;
    float time_write_pre_ = 0.0;

    double time_cur_;
    double time_pre_;
    double action_dt_accumulate_ = 0.0;

    Vector3_t base_lin_vel, base_ang_vel, base_ang_vel_lpf_;
    double heading;
    Eigen::Vector3d euler_angle_;

    // Joystick
    ros::NodeHandle nh_;

    void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
    ros::Subscriber joy_sub_;
    
    // Target velocity publisher for MuJoCo visualization
    ros::Publisher target_vel_pub_;

    std::string base_path = "";
    void loadCommand(const std::string &command_file);
    void updateCommandFromTimeline(const std::string &command_file);

    // BIPED WALKING PARAMETER
    float phase_indicator_ = 0;
    Eigen::Vector3d commands_;
    Eigen::Vector3d command_vel_filtered_;
    Eigen::Vector3d command_vel_filtered_prev_;
    double target_heading_;
    bool heading_mode_ = false;
    float step_period_ = 0.8;
    float step_ticks_ = 0.0;
    float max_stride_x = 0.4;
    float max_stride_y = 0.12;
    float max_stride_yaw = 0.4;

    float vel_scale_x_ = 0.6;
    float vel_scale_y_ = 0.2;

    int ctrl_mode = 2; // 0 for joystick, 1 for command file, 2 for timeline file

private:
    Eigen::VectorQd ControlVal_;
    unsigned int walking_tick = 0;
    unsigned int walking_tick_container = 0;

    bool cam_control_active_ = false;
    double cam_quiet_timer_us_ = 0.0;
    const double cam_release_duration_us_ = 0.15e6;
    const double command_filter_alpha_ = 0.25;
    const double command_change_threshold_ = 0.6;
    double arm_swing_phase_ = 0.0;

    Ort::Env env;
    Ort::Session session;
    Ort::Session session_n;
    Ort::Session session_dn;
    Ort::Session session_d;
    Ort::Session session_c;
    Ort::MemoryInfo memory_info;

};
