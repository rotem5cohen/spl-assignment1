#include <fstream>
#include <stdexcept>
#include "json.hpp"

#include "Server.h"

#include "Model.h"
#include "SchedulingPolicy.h"
#include <iostream>
#include <algorithm>
#include <utility>

using namespace std;
using json = nlohmann::json;

Config Server::readConfig(string configPath) const{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for(auto& gpu: config["gpus"]){
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for(auto& model: config["models"]){
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for(auto& request: config["requests"]){
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if(typeStr != "generative" && typeStr != "text_embedder"){
            throw invalid_argument("invalid request type");
        }
        RequestType type = typeStr=="generative"?RequestType::Generative:RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0) {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0) {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id, type, modelId, admissionCycle, prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0) {
        throw invalid_argument("missing scheduling policy");
    }
    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}

Server::Server(std::string configPath)
    : gpus(),
      models(),
      nonSubmittedRequests(),
      waitingRequests(),
      activeRequests(),
      policy(nullptr),
      cycle(0),
      activeGpuIndices() {

    Config config = readConfig(configPath);

    for (const GPUConfig& gpuConfig : config.gpus) {

        std::size_t blocksNum =
            gpuConfig.memoryBytes / config.blockSize;

        gpus.emplace_back(blocksNum, config.blockSize);
    }

    for (const ModelConfig& modelConfig : config.models) {

        models.push_back(
            new Model(
                modelConfig.id,
                modelConfig.embeddingSize
            )
        );
    }

    for (const RequestConfig& req : config.requests) {

    Model* model = findModel(req.modelId);

      if (req.type == RequestType::Generative) {

        nonSubmittedRequests.push_back(
            new GenerativeRequest(
                req.id,
                *model,
                req.admissionCycle,
                req.prompt,
                req.scriptedOutput,
                req.maxOutputTokens
            )
        );

      } else {

        nonSubmittedRequests.push_back(
            new EmbeddingRequest(
                req.id,
                *model,
                req.admissionCycle,
                req.prompt
            )
        );
      }
    }

    if (config.policy == "arrivalOrder") {
    policy = new ArrivalOrderPolicy();

    } else if (config.policy == "shortestJob") {
        policy = new ShortestJobPolicy();

    } else if (config.policy == "longestPrefix") {
        policy = new LongestPrefixPolicy();

    } else {
        throw std::invalid_argument("Invalid scheduling policy");
    }
}

Model* Server::findModel(const std::string& modelId) const {
    for (Model* model : models) {
        if (model->getModelId() == modelId) {
            return model;
        }
    }

    throw std::invalid_argument("Model not found");
}

void Server::clear() {

    for (Request* request : nonSubmittedRequests) {
        delete request;
    }

    for (Request* request : waitingRequests) {
        delete request;
    }

    for (Request* request : activeRequests) {
        delete request;
    }

    for (Model* model : models) {
        delete model;
    }

    delete policy;
    policy = nullptr;

    nonSubmittedRequests.clear();
    waitingRequests.clear();
    activeRequests.clear();
    activeGpuIndices.clear();
    models.clear();
    gpus.clear();

    cycle = 0;
}

Server::~Server() {
    clear();
}

Server::Server(const Server& other)
    : gpus(other.gpus),
      models(),
      nonSubmittedRequests(),
      waitingRequests(),
      activeRequests(),
      policy(nullptr),
      cycle(other.cycle),
      activeGpuIndices(other.activeGpuIndices) {

    try {
        models.reserve(other.models.size());

        for (Model* model : other.models) {
            models.push_back(new Model(*model));
        }

        nonSubmittedRequests.reserve(
            other.nonSubmittedRequests.size()
        );

        for (Request* request : other.nonSubmittedRequests) {
            nonSubmittedRequests.push_back(
                request->clone(
                    *findModel(request->getModel().getModelId())
                )
            );
        }

        waitingRequests.reserve(
            other.waitingRequests.size()
        );

        for (Request* request : other.waitingRequests) {
            waitingRequests.push_back(
                request->clone(
                    *findModel(request->getModel().getModelId())
                )
            );
        }

        activeRequests.reserve(
            other.activeRequests.size()
        );

        for (Request* request : other.activeRequests) {
            activeRequests.push_back(
                request->clone(
                    *findModel(request->getModel().getModelId())
                )
            );
        }

        if (other.policy != nullptr) {
            policy = other.policy->clone();
        }

    } catch (...) {
        clear();
        throw;
    }
}

Server& Server::operator=(const Server& other) {

    if (this == &other) {
        return *this;
    }

    clear();

    try {
        gpus = other.gpus;
        cycle = other.cycle;
        activeGpuIndices = other.activeGpuIndices;

        models.reserve(other.models.size());

        for (Model* model : other.models) {
            models.push_back(new Model(*model));
        }

        nonSubmittedRequests.reserve(
            other.nonSubmittedRequests.size()
        );

        for (Request* request : other.nonSubmittedRequests) {
            nonSubmittedRequests.push_back(
                request->clone(
                    *findModel(request->getModel().getModelId())
                )
            );
        }

        waitingRequests.reserve(
            other.waitingRequests.size()
        );

        for (Request* request : other.waitingRequests) {
            waitingRequests.push_back(
                request->clone(
                    *findModel(request->getModel().getModelId())
                )
            );
        }

        activeRequests.reserve(
            other.activeRequests.size()
        );

        for (Request* request : other.activeRequests) {
            activeRequests.push_back(
                request->clone(
                    *findModel(request->getModel().getModelId())
                )
            );
        }

        if (other.policy != nullptr) {
            policy = other.policy->clone();
        }

    } catch (...) {
        clear();
        throw;
    }

}

Server::Server(Server&& other) noexcept
    : gpus(std::move(other.gpus)),
      models(std::move(other.models)),
      nonSubmittedRequests(std::move(other.nonSubmittedRequests)),
      waitingRequests(std::move(other.waitingRequests)),
      activeRequests(std::move(other.activeRequests)),
      policy(other.policy),
      cycle(other.cycle),
      activeGpuIndices(std::move(other.activeGpuIndices)) {

    other.gpus.clear();
    other.models.clear();
    other.nonSubmittedRequests.clear();
    other.waitingRequests.clear();
    other.activeRequests.clear();
    other.activeGpuIndices.clear();

    other.policy = nullptr;
    other.cycle = 0;
}

Server& Server::operator=(Server&& other) noexcept {

    if (this == &other) {
        return *this;
    }

    clear();

    gpus = std::move(other.gpus);
    models = std::move(other.models);

    nonSubmittedRequests =
        std::move(other.nonSubmittedRequests);

    waitingRequests = std::move(other.waitingRequests);
    activeRequests = std::move(other.activeRequests);
    activeGpuIndices = std::move(other.activeGpuIndices);

    policy = other.policy;
    cycle = other.cycle;

    other.gpus.clear();
    other.models.clear();
    other.nonSubmittedRequests.clear();
    other.waitingRequests.clear();
    other.activeRequests.clear();
    other.activeGpuIndices.clear();

    other.policy = nullptr;
    other.cycle = 0;

    return *this;
}

