# bash.py
# Explanation of Bash
# Bash is a Unix shell and command language. It's used for scripting and command-line operations.
# In Python, we can execute Bash commands using the subprocess module.

import subprocess  # Import the subprocess module to run external commands
import sys  # Import sys module (though not used in this example, commonly needed for system operations)

def run_bash_command(command):  # Define a function to execute Bash commands safely
    """
    Execute a Bash command and return the output.
    """
    try:  # Try block to handle potential errors during command execution
        result = subprocess.run(command, shell=True, capture_output=True, text=True)  # Run the command with shell=True, capture output as text
        if result.returncode == 0:  # Check if the command executed successfully (return code 0 means success)
            return result.stdout.strip()  # Return the standard output, stripped of whitespace
        else:  # If the command failed (non-zero return code)
            return f"Error: {result.stderr.strip()}"  # Return the error message from stderr
    except Exception as e:  # Catch any exceptions that might occur
        return f"Exception: {str(e)}"  # Return the exception message as a string

# Examples of common Bash commands
print("1. List files in current directory:")  # Print a header for the first example
output = run_bash_command("ls -la")  # Execute the 'ls -la' command to list all files with details
print(output)  # Print the output of the command

print("\n2. Check current working directory:")  # Print a header for the second example
output = run_bash_command("pwd")  # Execute the 'pwd' command to show current directory path
print(output)  # Print the output of the command

print("\n3. Echo a message:")  # Print a header for the third example
output = run_bash_command("echo 'Hello from Bash!'")  # Execute the 'echo' command to display a message
print(output)  # Print the output of the command

print("\n4. Check system information:")  # Print a header for the fourth example
output = run_bash_command("uname -a")  # Execute the 'uname -a' command to show system information
print(output)  # Print the output of the command

# Bash scripting concepts:
# - Variables: var="value"; echo $var
# - Loops: for i in {1..5}; do echo $i; done
# - Conditionals: if [ condition ]; then ... fi
# - Functions: function_name() { commands; }
# - Pipes: command1 | command2
# - Redirection: command > file.txt