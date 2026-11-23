## Prompt
Implement the following idea. First create read in the attached files you will need. Then write out briefly an implementation plan that gaurantees all the requested feature details. Then implement this runner as a python file which will live in the repo root with the name: auto_runner.py
1. There are many different problem instance datasets we could target, we want to automate execution of a code idea on multiple instances
2. Format of instances
	1. aAAA_tTT_XX_bBBB_LLL
		1. AAA has "010" which is 10. Number of assets
		2. TT is "10" for 10 days
		3. XX is the seed which is either
			1. "orig"
			2. sXX
				1. Where XX is a binary string "01"
		4. LLL
			1. Is the Risk Lambda
			2. If not specified use the parameter lambda
3. In particular we should be able to
	1. Easily select which instances to run
		1. Selection Options:
			1. manual selection by passing CLI list of indexes as the instances are listed in the QUBOL table 
				1. This overwrites all parameters to set except for: Local
			2. Parameters to set:
				1. tests: Int / String
					1. Either give the name of the tests as given below or index them in the order given below
				2. Silent:
					1. Whether the terminals should be executed silently or if it should actually display the terminal windows!
						1. THIS SHOULD BE OS AGNOSTIC!
				3. Seed : String
					1. Choose one seed that is used in picking the instance for each test. Except for manual selection or test all
					2. Only use these for cases where "XX" is where the seed should be
				4. Local : Bool
					1. Wether to run locally or use the cloud (and credits)
				5. Risk : int
					1. Except for the risks tests where this gets overwritten
					2. 8 possibilities indexed by int
						1. Should be hardcoded array / switch case
						2. {0, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2}
					3. Default should be: 1e-4
			3. CompareToBest
				1. Only test instances which have a known best solution
				2. Instances:
					1. a010_t10_s02_b004_l1e-05
					2. p0_a10_t15_XX_b004
					3. a050_t10_s02_b020_l0
			4. CompareToOptimal
				1. Only test instances with a known optimum
				2. This overwrites the Seed parameter!
				3. Instances
					1. a010_t10_s02_b004_l1e-05
					2. a010_t15_orig_b004_l1e-06
					3. a050_t15_s02_b020_l0
			5. local_size_tests
				1. po_a010_t10_XX_b004
				2. po_a050_t10_XX_b020
				3. p0_a200_t10_XX_b050
			6. cloud_size_tests
				1. po_a050_t15_XX_b020
				2. po_a200_t15_XX_b050
				3. po_a400_t15_XX_b100
			7. local_small_risk_tests
				1. a010_t10_XX_b004_l1e-06
				2. a010_t10_XX_b004_l1e-04
				3. a010_t10_XX_b004_l1e-02
			8. local_big_risk_tests
				1. a050_t15_XX_b020_l1e-06
				2. a050_t15_XX_b020_l1e-04
				3. a050_t15_XX_b020_l1e-02
			9. cloud_small_risk_tests
				1. a200_t15_XX_b050_l1e-06
				2. a200_t15_XX_b050_l1e-04
				3. a200_t15_XX_b050_l1e-02
			10. cloud_big_risk_tests
				1. a400_t15_XX_b100_l1e-06
				2. a400_t15_XX_b100_l1e-04
				3. a400_t15_XX_b100_l1e-02
4. The wrapper auto runner should work like this
	1. It is a python file which is called per CLI with the above parameters given to appropriate flags
		1. The first argument should always be the name of the python file to execute
			1. This file expects as input the data from an instance and outputs the found solution
	2. Then the auto runner should
		1. IF LOCAL
			1. Check available resources
				1. Here you should check heuristically predict how much resources each instance task will occupy
				2. From notebook_01 for example
					1. "`po_a010_t10_orig` with $B = 4$ and $\lambda = 10^{-4}$: 710 variables, best-known value **−69 482**. With `USE_CLOUD = True` the 70 sub-programs of each run execute in parallel on Qoro's cloud. Each run sends 70 × 501 ≈ 35 000 circuits (lower `MC_POPULATION` or `MC_ITERATIONS` to spend fewer credits). Locally, each run takes under a minute, depending on your machine."
		2. IF CLOUD
			1. Check available credits
				1. If low on credits run one instance at a time
		3. Based on the available resources calculated start in parallel terminals executing the argument passed python file with one of the instances from the task stack
			1. It should create a dir to store all the .sol files and the final .json file for this auto run
				1. From the repo root target: /AutoData/..
					1. Create AutoData if it doesnt exist already
					2. the .. should be a new dir to store all the results
						1. Its name should be: the name of the test type + the hour + minutes when saving
			2. Each task should be passed with its terminal
				1. They should be launched within the newly created dir so the .sol files are created there
				2. The full name of the instance
				3. The path to the instance dir it needs to gather the covariance matrix and stock prices from
					1. BEWARE! This should be relative from the data dir it is executed from
				4. The parameters that uniquely define the instance
					1. Asset count, Days, Risk factor (lambda), budget (B)
				5. A Boolean CLOUD
					1. Wether to execute in cloud or locally
		4. The outputs from an instance task will be
			1. Indirectly
				1. It will generate in the data/dir a .sol file named after the instance
			2. Directly
				1. The score associated with the best .sol file
				2. An additional JSON object that contains some additional info which can change per run
		5. Analyse solution by running solution_checker.py (in the repo root) with as argument the solution file (.sol)
			1. The output should be stored in a dictionary to be serialized later and added to JSON file below
			2. The solution_checker.py will write to the terminal its outputs
				1. Analyse the file and ensure you correctly register these outputs
		6. Collect all the outputs and store them in a JSON file
			1. The filename should be: TYPE_TIME
				1. TYPE:
					1. The name of the test run
				2. TIME:
					1. The hour and minute all outputs were done
			2. You should never overwrite a json file. If a name conflict emerges just add an increment at the end until a unique filename is available
			3. The JSON format should contain this data
				1. The test name at the top
				2. A collection of the instance names run as strings
				3. For each instance a separate block containing:
					1. The start and end times of the related task
					2. The credits used if LOCAL = false
					3. All the output collected
						1. The best solution file name
						2. The associated benchmark score
					4. The best known score
						1. Extract this from "qoblib_instance_benchmarks.csv" in the root of the repository
							1. Look to the "Instance" and "Parameters" columns
							2. Read in the csv in question when designing the algorithm to ensure you match the code typing and formatting to the CSV ones
					5. Extra
						1. Some JSON object to add at the bottom of the file
							1. Its exact structure is not known