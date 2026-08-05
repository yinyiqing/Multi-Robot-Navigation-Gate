#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ros/ros.h>

#include <multi_robot_scenario/StepWorld.h>

namespace gazebo
{
class FixedStepWorldPlugin : public WorldPlugin
{
public:
  void Load(physics::WorldPtr world, sdf::ElementPtr) override
  {
    if (!ros::isInitialized())
    {
      gzerr << "FixedStepWorldPlugin requires gazebo_ros_api_plugin" << std::endl;
      return;
    }

    this->world = std::move(world);
    this->node.reset(new ros::NodeHandle());
    this->service = this->node->advertiseService(
        "/gazebo/step_world", &FixedStepWorldPlugin::StepWorld, this);
    gzmsg << "Fixed-step world service ready at /gazebo/step_world" << std::endl;
  }

private:
  bool StepWorld(
      multi_robot_scenario::StepWorld::Request &request,
      multi_robot_scenario::StepWorld::Response &response)
  {
    std::lock_guard<std::mutex> guard(this->stepMutex);
    if (!this->world)
    {
      response.success = false;
      response.status_message = "Gazebo world is unavailable";
      return true;
    }
    if (request.steps == 0)
    {
      response.success = false;
      response.status_message = "Step count must be positive";
      response.sim_time = this->world->SimTime().Double();
      return true;
    }
    if (!this->world->IsPaused())
    {
      response.success = false;
      response.status_message = "Gazebo must be paused before fixed stepping";
      response.sim_time = this->world->SimTime().Double();
      return true;
    }

    const double before = this->world->SimTime().Double();
    this->world->Step(request.steps);
    const double after = this->world->SimTime().Double();
    const double expected =
        static_cast<double>(request.steps) *
        this->world->Physics()->GetMaxStepSize();
    const double tolerance =
        std::max(1e-9, 0.5 * this->world->Physics()->GetMaxStepSize());

    response.sim_time = after;
    if (std::abs((after - before) - expected) > tolerance)
    {
      response.success = false;
      response.status_message =
          "Gazebo completed an unexpected amount of simulation time";
      return true;
    }

    response.success = true;
    response.status_message = "ok";
    return true;
  }

  physics::WorldPtr world;
  std::unique_ptr<ros::NodeHandle> node;
  ros::ServiceServer service;
  std::mutex stepMutex;
};

GZ_REGISTER_WORLD_PLUGIN(FixedStepWorldPlugin)
}  // namespace gazebo
