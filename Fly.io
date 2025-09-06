# This is the main configuration file for your app on Fly.io
app = "econo-backend" # You will set this in Part 2
primary_region = "iad" # Example: Ashburn, VA. Choose a region near your users.

# This tells Fly.io to build your app using the Dockerfile
[build]
  dockerfile = "Dockerfile"

# This section configures how your app is served over the internet
[http_service]
  internal_port = 8080 # This must match the port in the Dockerfile's CMD
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 256