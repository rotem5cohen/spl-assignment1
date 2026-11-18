#ifndef REQUEST_H_
#define REQUEST_H_

#include <string>
#include <vector>

class Model;
class GPU;

class Request{
protected:
    int id;
    const Model& model;
    int admissionCycle;
    std::vector<std::string> tokens;
    std::vector<std::size_t> memoryBlocks;
    std::size_t reservedBlocks;
public:
    Request(int id, const Model& model, int admissionCycle, std::string prompt);
    virtual std::size_t getNeededMemory(const GPU& gpu) const=0;
    virtual std::size_t getMaxNeededMemory() const=0;
    virtual std::vector<std::string> getCacheableTokens() const=0;
    virtual void allocateOneBlock(GPU& gpu, std::size_t slot) const = 0;
    virtual Request* clone(const Model& model) const = 0;
    void deallocateMemory(GPU& gpu);
    virtual void act(GPU& gpu)=0;
    virtual bool finished() const=0;
    virtual void reserveMemory(GPU& gpu) = 0;
    virtual std::string getOutput() const = 0;
    virtual std::size_t getNeedToUnreserve() const = 0;
    int getId() const;
    const Model& getModel() const;
    int getAdmissionCycle() const;
    virtual ~Request();
};

class GenerativeRequest: public Request{
private:
    std::string scriptedOutput;
    std::size_t maxOutputTokens;
    std::vector<std::string> outputTokens;
    std::size_t processedTokens;
    std::size_t generatedTokens;
public:
    GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt, std::string scriptedOutput,
    std::size_t maxOutputTokens);
    bool finished() const override;
    std::size_t getMaxNeededMemory() const override;
    std::size_t getNeededMemory(const GPU& gpu) const override;
    std::vector<std::string> getCacheableTokens() const override;
    void reserveMemory(GPU& gpu) override;
    std::size_t getNeedToUnreserve() const override;
    void act(GPU& gpu) override;
    void allocateOneBlock(GPU& gpu, std::size_t slot) const override;
    std::string getOutput() const override;
    Request* clone(const Model& model) const override;

};

class EmbeddingRequest: public Request{
private:
    std::size_t processedTokens;
    std::vector<int> output;
public:
    EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt);
    bool finished() const override;
    std::size_t getMaxNeededMemory() const override;
    std::size_t getNeededMemory(const GPU& gpu) const override;
    std::vector<std::string> getCacheableTokens() const override;
    void allocateOneBlock(GPU& gpu, std::size_t slot) const override;
    void reserveMemory(GPU& gpu) override;
    std::size_t getNeedToUnreserve() const override;
    void act(GPU& gpu) override;
    std::string getOutput() const override;
    Request* clone(const Model& model) const override;
};

#endif