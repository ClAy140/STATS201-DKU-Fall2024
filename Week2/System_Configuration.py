import subprocess

def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

# Gather system information
report = "### System Configuration Report\n\n"

# CPU Information
report += "**CPU Information:**\n" + run_command("sysctl -a | grep 'machdep.cpu'") + "\n\n"

# GPU Information
gpu_info = run_command("system_profiler SPDisplaysDataType | grep 'Chipset Model'")
report += "**GPU Information:**\n" + (gpu_info if gpu_info else "No GPU information found") + "\n\n"

# Memory Information
memory_info = run_command("vm_stat | awk '/free/ {free=$3} /active/ {active=$3} /inactive/ {inactive=$3} /wired/ {wired=$4} END {print \"Free: \" free*4096/1024/1024 \" MB\"; print \"Active: \" active*4096/1024/1024 \" MB\"; print \"Inactive: \" inactive*4096/1024/1024 \" MB\"; print \"Wired: \" wired*4096/1024/1024 \" MB\"}'")
report += "**Memory Information:**\n" + memory_info + "\n\n"

# Disk Space
report += "**Disk Space:**\n" + run_command("df -h") + "\n\n"

# Python Version
report += "**Python Version:**\n" + run_command("python3 --version") + "\n\n"

# Save system configuration report to a text file
with open("system_configuration_report.txt", "w") as file:
    file.write(report)

print("System configuration report saved to 'system_configuration_report.txt'")

# Save installed packages to a separate file
subprocess.run("pip3 list > installed_packages.txt", shell=True)
print("Installed packages list saved to 'installed_packages.txt'")
