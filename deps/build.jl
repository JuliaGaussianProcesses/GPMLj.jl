using PyCall

println("Running build.jl for the GPMLj package.")

# Change that to whatever packages you need.
const PACKAGES = ["gpflow==2.0.5", "tensorflow>=2.2", "tensorflow-probability>=0.9"]

# Use eventual proxy info
proxy_arg=String[]
if haskey(ENV, "http_proxy")
    push!(proxy_arg, "--proxy")
    push!(proxy_arg, ENV["http_proxy"])
end

# Import pip
try
    pyimport("pip")
catch
    # If it is not found, install it
    println("Pip not found on your system. Downloading it.")
    get_pip = joinpath(dirname(@__FILE__), "get-pip.py")
    download("https://bootstrap.pypa.io/get-pip.py", get_pip)
    run(`$(PyCall.python) $(proxy_arg) $get_pip --user`)
end

println("Installing required python packages using pip")
run(`$(PyCall.python) $(proxy_arg) -m pip install --upgrade pip setuptools`)
run(`$(PyCall.python) $(proxy_arg) -m pip install $(PACKAGES)`)