FROM julia:1.11-bookworm

WORKDIR /app
COPY Project.toml Manifest.toml ./
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

COPY . .

ENTRYPOINT ["julia", "--project=."]
CMD ["main.jl"]
