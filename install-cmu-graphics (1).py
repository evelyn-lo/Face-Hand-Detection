# Version 2.3
import platform
import subprocess
import os
import sys

def getPackageVersion(packageName):
    # Returns the version of the package, or returns None if not installed

    # Get installed packages with their versions
    package_versions = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    package_versions = package_versions.decode('utf-8')
    
    for package in package_versions.splitlines():
        name, version = package.split('==')
        if name == packageName:
            return version
        
    return None

def install(packageName):
    # Installs a package, checking and printing the results

    # <current_python> -m pip install package
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', packageName])

    # Check that the package is installed
    packageVersion = getPackageVersion(packageName)
    if packageVersion is None:
        print(f'Failed to install {packageName}. Please capture the output in the terminaland send to an instructor.')
    else:
        print(f'Success: {packageName} is installed (version {packageVersion})')

def checkXcodeCLT():
    try:
        print("Checking if xcode command line tools are installed...")
        output = subprocess.check_output(["xcode-select", "-p"], text=True)
        print(f"Succesfully found xcode command line tools, version {output}")
        return True
    except FileNotFoundError:
        print("xcode command line tools are not installed")
        print()
        print("1) Open the Terminal application (in Applications/Utilities)")
        print("2) In the Terminal type: xcode-select --install")
        print('\tIf the above step results in: "xcode-select: note: install requested for command line developer tools", then:')
        print("\t2a) Open System Settings (available from apple menu in top-left of screen); go to General->Software Update; click on Update Now next to Updates Available: Command Line Tools for Xcode; and follow those installation prompts.")
        print('\t(After it installs, the "Update Now" may still be available, but it likely successfuly installed)')
        print()
        print('When finished installing xcode command line tools, re-run this install-cmu-graphics.py file for the next set of instructions')
        print()
        print("Definitely let us know if you need any help with this!")
        return False

def checkBrew():
    try:
        print("Checking if brew is installed...")
        output = subprocess.check_output(["brew", "--version"], text=True)
        print(f"Succesfully found brew, version {output}")
        return True
    
    except FileNotFoundError:
        print("brew is not installed")
        print()
        # print("Install the brew package manager by following the instructions at https://brew.sh/")
        # print("Definitely let us know if you need any help with this!")
        # print()
        # print("Once brew is installed, run this file again to complete the installation of cmu-graphics")
        return False

def installBrew():
    print("Attempting to install brew...")
    output = subprocess.check_output(['curl', '-fsSL', 'https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh'], text=True)
    with open('install_brew.sh', 'w') as f:
        f.write(output)

    subprocess.check_call(['/bin/bash', './install_brew.sh'], text=True)
    print(f"Completed brew installation")
    print()

    print("Adding brew to your PATH...")

    home_dir = os.path.expanduser('~')
    if os.environ['SHELL'] == '/bin/bash':
        profile_file = home_dir + "/.bash_profile"
    else:
        profile_file = home_dir + "/.zprofile"

    try:    
        output = subprocess.check_output(['/usr/local/bin/brew', 'shellenv'], text=True)
    except FileNotFoundError:
        output = subprocess.check_output(['/opt/homebrew/bin/brew', 'shellenv'], text=True)

    with open(profile_file, 'a') as f:
        f.write('\n\n'+output)

    print('Completed installation of brew and added brew to your PATH')

def installCairoCustom():
    # Assuming brew is already installed
    # Includes installing brew pkg-config and pycairo

    brew_list = subprocess.check_output(["brew", "list"], text=True)
    cairoVersion = None
    if "cairo" in brew_list:
        cairoVersion = subprocess.check_output(['brew', 'list', 'cairo', '--version'], text=True)
        print(f'Found cairo version {cairoVersion}')

        if '1.16.0_5' not in cairoVersion:
            if getPackageVersion('pycairo') == '1.25.0':
                subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'cmu-graphics'])
                subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'pycairo'])

            print(f'Attempting to remove cairo version {cairoVersion}...')
            subprocess.check_call(['brew', 'remove', 'cairo'], text=True)
            cairoVersion = None

    if cairoVersion is None:
        print("Attempting custom brew install of cairo version 1.16.0_5...")

        output = subprocess.check_output(['curl', 'https://raw.githubusercontent.com/Homebrew/homebrew-core/f6f14de3fd98b04c5967d37a960990b73ccf6af7/Formula/c/cairo.rb'], text=True)

        cairofile = 'cairo.rb'
        with open(cairofile, 'w') as f:
            f.write(output)

        try:
            subprocess.check_call(['brew', 'install', cairofile], text=True)
        except subprocess.CalledProcessError as e:
            print(f"The custom brew install probably return a nonzero status even if it was successful. Here is the code, just in case: {e.returncode}")
            print("Carrying on with the installation...")


    if "pkg-config" not in brew_list:
        print("Installing brew package pkg-config...")
        subprocess.check_call(['brew', 'install', 'pkg-config'], text=True)
        print("Successfully installed brew package pkg-config")

    print("Installing pycairo, specifically version 1.24.0...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pycairo==1.24.0'])
    print("Successfully installed pycairo version 1.24.0")

def installMac():
    print("It looks like your are running Mac OSX...")
    print()
    if not checkXcodeCLT():
        return False

    if not checkBrew():
        installBrew()

    installCairoCustom()

    print('Installing cmu-graphics on Mac OSX...')
    install('cmu-graphics')
    print()
    print('DONE')

def installWindows():
    print('Installing cmu-graphics on Windows...')
    install('cmu-graphics')
    print()
    print('DONE')

def installLinux():
    print("It looks like you are likely running Linux and thus are probably good at following instuctions. Make sure to follow the instructions at this link before pip installing cmu-graphics: https://pycairo.readthedocs.io/en/latest/getting_started.html")

    response = input("Have you alread installed Pycairo and want to continue to installing cmu-graphics? (Y/n): ")
    if response in ('y', 'Y', 'yes', 'Yes', 'YES', ''):
        print('Installing cmu-graphics on Linux...')
        install('cmu-graphics')
        print()
        print('DONE')
    else:
        print("Please install Pycairo and then run this file again to install cmu-graphics.")
        print("Pycairo installation instructions can be found at https://pycairo.readthedocs.io/en/latest/getting_started.html")

# result = platform.system()
# print(type(result))
# print(result)

try:
    if platform.system() == "Darwin":
        installMac()
    elif platform.system() == "Windows":
        installWindows()
    else:
        installLinux()

except subprocess.CalledProcessError as e:
    print(f"Command failed with return code {e.returncode}")
    print("Hmmm, something unexpected went wrong.")
    print("Please come get help on Piazza or in office hours.")
    exit(-1)
