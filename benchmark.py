import argparse
from vllm import LLM, SamplingParams
import time

llm_map = {
    "llama": "meta-llama/Meta-Llama-3.1-8B",
    "opt": "facebook/opt-125m"
}

prompt = ["""Hi, tell me a piece of useful and actionable advice on triton and GPU programming."""]

sampling_params = SamplingParams(temperature=0.8, top_p = 0.95)

def main(args):
    # Initialize the vLLM engine.
    llms = args.LLMs
    n = args.num_trials
    if llms[0] == "all":
        llms = llm_map.keys()

    for llm in llms:
        llm_name = llm_map[llm] 
        print(f"Testing {llm_name} LLM speed. ")
        llm = LLM(model=llm_name)
        start_time = time.time()
        prompts = prompt * n
        assert len(prompts) == n
        outputs = llm.generate(prompt, sampling_params)

        for output in outputs:
            ptext = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {ptext!r}, Generated text: {generated_text!r}")
        end_time = time.time()
        print(f"The total time used was: {(end_time-start_time):.3f}\n\n")



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_trials", default=100, type=int, help="Defines the number of trials to test the LLM")
    parser.add_argument("-l", "--LLMs", nargs="+", help="Pass in a list of LLMs to try. \"all\" will automatically run all LLMs.")
    args = parser.parse_args()
    main(args)


