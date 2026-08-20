function save_dataset(path::AbstractString, data::IsingDataset)
    open(path, "w") do io
        serialize(io, data)
    end
    return path
end

function load_dataset(path::AbstractString)
    open(path, "r") do io
        return deserialize(io)
    end
end
