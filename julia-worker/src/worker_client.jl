# HTTP client for communicating with the FastAPI backend
# All worker endpoints are under /worker/*

using HTTP
using JSON3

mutable struct WorkerClient
    base_url::String
end

WorkerClient(; host::String="localhost", port::Int=5000) =
    WorkerClient("http://$host:$port")

function _url(client::WorkerClient, path::String)
    client.base_url * "/worker" * path
end

function _get(client::WorkerClient, path::String)
    resp = HTTP.get(_url(client, path); status_exception=false)
    resp.status == 404 && return nothing
    s = String(resp.body)
    s == "" && return nothing
    JSON3.read(s)
end

function _post(client::WorkerClient, path::String; body=nothing)
    headers = ["Content-Type" => "application/json"]
    if body === nothing
        resp = HTTP.post(_url(client, path); status_exception=false)
    else
        resp = HTTP.post(_url(client, path); headers, body=JSON3.write(body), status_exception=false)
    end
    resp.status == 404 && return nothing
    s = String(resp.body)
    isempty(s) && return nothing
    JSON3.read(s)
end

function _put(client::WorkerClient, path::String; body=nothing, query=nothing)
    headers = ["Content-Type" => "application/json"]
    kw = Dict{Symbol,Any}(:status_exception => false, :headers => headers)
    body !== nothing && (kw[:body] = JSON3.write(body))
    query !== nothing && (kw[:query] = query)
    resp = HTTP.put(_url(client, path); kw...)
    resp.status == 404 && return nothing
    s = String(resp.body)
    isempty(s) && return nothing
    JSON3.read(s)
end

function _delete(client::WorkerClient, path::String)
    HTTP.delete(_url(client, path); status_exception=false)
    nothing
end

# --- Jobs ---

function get_jobs(client::WorkerClient)
    _get(client, "/jobs")
end

function launch_job(client::WorkerClient, description::String)
    _post(client, "/jobs/$(HTTP.URIs.escapeuri(description))/launch")
end

function get_job_status(client::WorkerClient, description::String)::Union{Int, Nothing}
    result = _get(client, "/jobs/$(HTTP.URIs.escapeuri(description))/status")
    result === nothing ? nothing : Int(result.status)
end

function update_timing(client::WorkerClient, description::String, elapsed::Int, remaining::Int)
    _put(client, "/jobs/$(HTTP.URIs.escapeuri(description))/timing";
         query=Dict("elapsed" => elapsed, "remaining" => remaining))
end

function update_alpha(client::WorkerClient, description::String, alpha::Float64)
    _put(client, "/jobs/$(HTTP.URIs.escapeuri(description))/alpha";
         query=Dict("alpha" => alpha))
end

function delete_job(client::WorkerClient, description::String)
    _delete(client, "/jobs/$(HTTP.URIs.escapeuri(description))")
end

# --- Agents ---

function get_agent(client::WorkerClient, name::String)
    _get(client, "/agents/$(HTTP.URIs.escapeuri(name))")
end

function update_agent(client::WorkerClient, name::String, updates::Dict)
    _put(client, "/agents/$(HTTP.URIs.escapeuri(name))"; body=updates)
end

# --- Weights (binary upload/download) ---

function upload_weights(client::WorkerClient, name::String, local_path::String)
    url = _url(client, "/agents/$(HTTP.URIs.escapeuri(name))/weights")
    open(local_path, "r") do io
        data = read(io)
        form = HTTP.Form(Dict("file" => HTTP.Multipart("weights.bin", IOBuffer(data), "application/octet-stream")))
        HTTP.post(url, [], form; status_exception=false)
    end
end

function download_weights(client::WorkerClient, name::String, local_path::String)::Bool
    url = _url(client, "/agents/$(HTTP.URIs.escapeuri(name))/weights")
    resp = HTTP.get(url; status_exception=false)
    resp.status == 404 && return false
    open(local_path, "w") do io
        write(io, resp.body)
    end
    true
end

# --- Games ---

function save_game(client::WorkerClient, game_data::Dict)
    _post(client, "/games"; body=game_data)
end

function update_watch_game(client::WorkerClient, user::String, moves::Vector, tiles::Vector)
    _put(client, "/games/$(HTTP.URIs.escapeuri(user))/moves";
         body=Dict("moves" => moves, "tiles" => tiles))
end

function delete_game(client::WorkerClient, user::String)
    _delete(client, "/games/$(HTTP.URIs.escapeuri(user))")
end

# --- Logs ---

function add_log(client::WorkerClient, user::String, text::String)
    _post(client, "/logs"; body=Dict("userName" => user, "text" => text))
end

# --- Watch ---

function set_watch_loading(client::WorkerClient, description::String, loading::Bool)
    _put(client, "/watch/$(HTTP.URIs.escapeuri(description))/loading";
         query=Dict("loading" => loading))
end

# --- Cleanup ---

function cleanup(client::WorkerClient)
    _post(client, "/cleanup")
end
