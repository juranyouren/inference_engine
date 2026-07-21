import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams
from utils.public_functions import vllm_invoke, save_json
from llm_inference.prompts import (
    ROOT_CAUSE_META, ROOT_CAUSE_Meta_thinker, ROOT_CAUSE_Reasoner,
    ROOT_CAUSE_Reasoner1, ROOT_CAUSE_Verifier,
    ROOT_CAUSE_Reasoner_Coopetiton, ROOT_CAUSE_Meta_thinker_Coopetition,
    ROOT_CAUSE_Verifier_Coopetiton, ROOT_CAUSE_NAIVE, ROOT_CAUSE_FEW_SHOT,
    BGP_FEW_SHOT, NETWORK_FEW_SHOT, INTERFACE_FEW_SHOT, INBW_FEW_SHOT,
)
import json

class RCAGenerator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.llm = None
        self._init_llm()
    
    def _init_llm(self):
        """初始化LLM模型"""
        self.llm = LLM(
            model=self.model_path, 
            tensor_parallel_size=len(os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(',')),
            trust_remote_code=True,
            gpu_memory_utilization=0.9, 
            max_model_len=16384
        )
        print(f"LLM模型已初始化: {self.model_path}")

    def generate_rca_analysis(self, alarm_data_list, output_dir,batch_indices):
        """
        批量生成根因分析
        
        Args:
            alarm_data_list: 告警数据字典列表
            output_dir: 输出目录路径
            
        Returns:
            analysis_results: 分析结果列表
        """
        print(f"正在批量生成根因分析，共{len(alarm_data_list)}个case...")
        
        # 构建所有输入提示词
        input_prompts = []
        for alarm_data in alarm_data_list:
            input_prompt = ROOT_CAUSE_META.format(
                alarm_type=alarm_data['alarm_type'],
                semantic_labels=json.dumps(alarm_data['semantic_labels'], ensure_ascii=False, indent=2),
                alarm_time=alarm_data["alarm_time"],
                sop=alarm_data['sop'],
                # sop='',
                root_cause_candidates=json.dumps(alarm_data['root_cause_candidates'], ensure_ascii=False, indent=2)
            )
            input_prompts.append(input_prompt)
        
        # 批量生成根因分析
        responses = vllm_invoke(
            self.llm,
            inputs=input_prompts,
            sampling_params=SamplingParams(temperature=0.5, top_p=0.9, max_tokens=4096),
            batch_size=16  
        )
        
        # 保存结果（如果提供了输出目录）
        if output_dir:
            self._save_results_meta(responses, output_dir,batch_indices)
        
        return responses
    def generate_rca_analysis_fs(self, alarm_type,alarm_data_list, output_dir,batch_indices):
        """
        批量生成根因分析
        
        Args:
            alarm_data_list: 告警数据字典列表
            output_dir: 输出目录路径
            
        Returns:
            analysis_results: 分析结果列表
        """
        print(f"正在批量生成根因分析，共{len(alarm_data_list)}个case...")
        
        # 构建所有输入提示词
        input_prompts = []
        for alarm_data in alarm_data_list:
            input_prompt = ROOT_CAUSE_NAIVE.format(
                alarm_type=alarm_data['alarm_type'],
                semantic_labels=json.dumps(alarm_data['semantic_labels'], ensure_ascii=False, indent=2),
                alarm_time=alarm_data['alarm_time'],
                sop=alarm_data['sop'],
                # sop='',
                root_cause_candidates=json.dumps(alarm_data['root_cause_candidates'], ensure_ascii=False, indent=2)
            )
            if alarm_type=="BGP邻接状态改变": input_prompt=input_prompt+"示例："+BGP_FEW_SHOT
            if alarm_type=="网络设备掉线": input_prompt=input_prompt+"示例："+NETWORK_FEW_SHOT
            if alarm_type=="接口状态DOWN": input_prompt=input_prompt+"示例："+INTERFACE_FEW_SHOT
            if alarm_type=="入方向带宽利用率过高": input_prompt=input_prompt+"示例："+INBW_FEW_SHOT
            # if alarm_type=="BGP邻接状态改变": input_prompt=input_prompt+"示例："+NETWORK_FEW_SHOT+INTERFACE_FEW_SHOT+INBW_FEW_SHOT
            # if alarm_type=="网络设备掉线": input_prompt=input_prompt+"示例："+BGP_FEW_SHOT+INTERFACE_FEW_SHOT+INBW_FEW_SHOT
            # if alarm_type=="接口状态DOWN": input_prompt=input_prompt+"示例："+BGP_FEW_SHOT+NETWORK_FEW_SHOT+INBW_FEW_SHOT
            # if alarm_type=="入方向带宽利用率过高": input_prompt=input_prompt+"示例："+BGP_FEW_SHOT+INTERFACE_FEW_SHOT+NETWORK_FEW_SHOT
            input_prompts.append(input_prompt)
        
        # 批量生成根因分析
        responses = vllm_invoke(
            self.llm,
            inputs=input_prompts,
            sampling_params=SamplingParams(temperature=0.5, top_p=0.9, max_tokens=4096),
            batch_size=16  
        )
        
        # 保存结果（如果提供了输出目录）
        if output_dir:
            self._save_results_meta(responses, output_dir,batch_indices)
        
        return responses


    def generate_rca_analysis_meta(self, alarm_data_list, output_dirs=None, batch_indices=None):
        """
        批量生成Meta根因分析
        
        Args:
            alarm_data_list: 告警数据字典列表
            output_dirs: 输出目录路径列表（与alarm_data_list一一对应）
            batch_indices: 对应的case序号列表
            
        Returns:
            responses: 分析结果列表
        """
        print(f"正在批量生成Meta根因分析，共{len(alarm_data_list)}个case...")
        
        # 构建所有输入提示词
        input_prompts = []
        for alarm_data in alarm_data_list:
            input_prompt = ROOT_CAUSE_Meta_thinker.format(
                Last_round_outputs=alarm_data['Last_round_outputs'],
                alarm_type=alarm_data['alarm_type'],
                semantic_labels_key=json.dumps(alarm_data['semantic_labels_key'], ensure_ascii=False, indent=2),
                sop=alarm_data['sop'],
                # sop="",
                root_cause_candidates=json.dumps(alarm_data['root_cause_candidates'], ensure_ascii=False, indent=2)
            )
            input_prompts.append(input_prompt)
        
        # 批量生成根因分析
        responses = vllm_invoke(
            self.llm,
            inputs=input_prompts,
            sampling_params=SamplingParams(temperature=0.5, top_p=0.9, max_tokens=4096),
            batch_size=16,
            prompt_output_paths=(
                [os.path.join(output_dir, "prompt.txt") for output_dir in output_dirs]
                if output_dirs else None
            ),
        )
        
        # 保存结果（如果提供了输出目录和索引）
        if output_dirs and batch_indices:
            if len(output_dirs) == len(responses) and len(batch_indices) == len(responses):
                self._save_results(responses, output_dirs, batch_indices)
            else:
                print(f"警告: 目录数({len(output_dirs)})、索引数({len(batch_indices)})与响应数({len(responses)})不匹配")
        
        return responses
    
    def generate_rca_analysis_reasoner(self, alarm_data_list, output_dirs=None, batch_indices=None):
        """
        批量生成Reasoner根因分析
        
        Args:
            alarm_data_list: 告警数据字典列表
            output_dirs: 输出目录路径列表（与alarm_data_list一一对应）
            batch_indices: 对应的case序号列表
            
        Returns:
            responses: 分析结果列表
        """
        print(f"正在批量生成Reasoner根因分析，共{len(alarm_data_list)}个case...")
        
        # 构建所有输入提示词
        input_prompts = []
        for alarm_data in alarm_data_list:
            input_prompt = ROOT_CAUSE_Reasoner.format(
                meta_output=alarm_data['meta_output'],
                alarm_type=alarm_data['alarm_type'],
                alarm_time=alarm_data['alarm_time'],
                semantic_labels=json.dumps(alarm_data['semantic_labels'], ensure_ascii=False, indent=2),
                root_cause_candidates=json.dumps(alarm_data['root_cause_candidates'], ensure_ascii=False, indent=2)
            )
            input_prompts.append(input_prompt)
        
        # 批量生成根因分析
        responses = vllm_invoke(
            self.llm,
            inputs=input_prompts,
            sampling_params=SamplingParams(temperature=0.5, top_p=0.9, max_tokens=4096),
            batch_size=16,
            prompt_output_paths=(
                [os.path.join(output_dir, "prompt.txt") for output_dir in output_dirs]
                if output_dirs else None
            ),
        )
        
        # 保存结果（如果提供了输出目录和索引）
        if output_dirs and batch_indices:
            if len(output_dirs) == len(responses) and len(batch_indices) == len(responses):
                self._save_results(responses, output_dirs, batch_indices)
            else:
                print(f"警告: 目录数({len(output_dirs)})、索引数({len(batch_indices)})与响应数({len(responses)})不匹配")
        
        return responses
    def generate_rca_analysis_competition_batch(self, alarm_data_list, output_dirs=None):
        """
        批量使用相同提示词模板，不同采样参数运行三次
        """
        print(f"批量使用相同提示词模板进行三次分析，共{len(alarm_data_list)}个case...")
        
        # 定义三个不同的采样参数组合
        sampling_configs = [
            {"temperature": 0.1, "top_p": 0.9, "name": "Reasoner1"},
            {"temperature": 0.5, "top_p": 0.95, "name": "Reasoner2"},
            {"temperature": 0.8, "top_p": 1.0, "name": "Reasoner3"}
        ]
        
        # 为每个case准备存储
        all_responses_by_case = [[] for _ in range(len(alarm_data_list))]
        
        # 为每个采样配置批量处理
        for config_idx, config in enumerate(sampling_configs):
            reasoner_name = config['name']
            print(f"\n运行 {reasoner_name} 批次...")
            print(f"  参数: temperature={config['temperature']}, top_p={config['top_p']}")
            
            # 构建本批次所有case的提示词
            input_prompts = []
            for alarm_data in alarm_data_list:
                input_prompt = ROOT_CAUSE_Reasoner1.format(
                    alarm_type=alarm_data['alarm_type'],
                    semantic_labels=json.dumps(alarm_data['semantic_labels'], ensure_ascii=False, indent=2),
                    alarm_time=alarm_data['alarm_time'],
                    # sop="",
                    sop=alarm_data['sop'],
                    root_cause_candidates=json.dumps(alarm_data['root_cause_candidates'], ensure_ascii=False, indent=2)
                )
                input_prompts.append(input_prompt)
            
            # 批量调用大模型
            responses = vllm_invoke(
                self.llm,
                inputs=input_prompts,
                sampling_params=SamplingParams(
                    temperature=config['temperature'],
                    top_p=config['top_p'],
                    max_tokens=4096
                ),
                batch_size=16,
                prompt_output_paths=(
                    [
                        os.path.join(output_dir, f"prompt_{reasoner_name}.txt")
                        for output_dir in output_dirs
                    ]
                    if output_dirs else None
                ),
            )
            
            # 将响应按case存储
            for i, response in enumerate(responses):
                if i < len(all_responses_by_case):
                    all_responses_by_case[i].append(response)
        
        # 合并每个case的所有响应
        merged_responses = []
        for case_responses in all_responses_by_case:
            merged_response = ""
            for i, resp in enumerate(case_responses):
                merged_response += f"=== {sampling_configs[i]['name']} ===\n{resp}\n\n{'='*50}\n\n"
            merged_responses.append(merged_response)
        
        # 保存结果
        if output_dirs and len(output_dirs) == len(merged_responses):
            for i, (response, output_dir) in enumerate(zip(merged_responses, output_dirs)):
                if output_dir:
                    self._save_results_1([response], output_dir, [i])
        
        return merged_responses

    def generate_rca_analysis_verifier_batch(self, alarm_data_list, output_dirs=None):
        """
        批量生成Verifier分析
        """
        print(f"批量生成Verifier分析，共{len(alarm_data_list)}个case...")
        
        # 构建所有输入提示词
        input_prompts = []
        for alarm_data in alarm_data_list:
            input_prompt = ROOT_CAUSE_Verifier.format(
                alarm_type=alarm_data['alarm_type'],
                semantic_labels=json.dumps(alarm_data['semantic_labels'], ensure_ascii=False, indent=2),
                # sop='',
                sop=alarm_data['sop'],
                root_cause_candidates=json.dumps(alarm_data['root_cause_candidates'], ensure_ascii=False, indent=2),
                reasoner_outputs=alarm_data['reasoner_outputs']
            )
            input_prompts.append(input_prompt)
        
        # 批量生成Verifier分析
        responses = vllm_invoke(
            self.llm,
            inputs=input_prompts,
            sampling_params=SamplingParams(temperature=0.5, top_p=0.9, max_tokens=4096),
            batch_size=16,
            prompt_output_paths=(
                [os.path.join(output_dir, "prompt.txt") for output_dir in output_dirs]
                if output_dirs else None
            ),
        )
        
        # 保存结果
        if output_dirs and len(output_dirs) == len(responses):
            for i, (response, output_dir) in enumerate(zip(responses, output_dirs)):
                if output_dir:
                    self._save_results_1([response], output_dir, [i])
        
        return responses
    def generate_rca_analysis_meta_coopetiton_batch(self, alarm_data_list, output_dirs=None):
        """
        批量生成Coopetition Meta根因分析
        
        Args:
            alarm_data_list: 告警数据字典列表
            output_dirs: 输出目录路径列表（与alarm_data_list一一对应）
            
        Returns:
            responses: 分析结果列表
        """
        print(f"批量生成Coopetition Meta根因分析，共{len(alarm_data_list)}个case...")
        
        # 构建所有输入提示词
        input_prompts = []
        for alarm_data in alarm_data_list:
            input_prompt = ROOT_CAUSE_Meta_thinker_Coopetition.format(
                Last_round_outputs=alarm_data['Last_round_outputs'],
                alarm_type=alarm_data['alarm_type'],
                semantic_labels_key=json.dumps(alarm_data['semantic_labels_key'], ensure_ascii=False, indent=2),
                sop=alarm_data['sop'],
                # sop="",
                root_cause_candidates=json.dumps(alarm_data['root_cause_candidates'], ensure_ascii=False, indent=2)
            )
            input_prompts.append(input_prompt)
        
        # 批量生成根因分析
        responses = vllm_invoke(
            self.llm,
            inputs=input_prompts,
            sampling_params=SamplingParams(temperature=0.5, top_p=0.9, max_tokens=4096),
            batch_size=16,
            prompt_output_paths=(
                [os.path.join(output_dir, "prompt.txt") for output_dir in output_dirs]
                if output_dirs else None
            ),
        )
        
        # 保存结果
        if output_dirs and len(output_dirs) == len(responses):
            for i, (response, output_dir) in enumerate(zip(responses, output_dirs)):
                if output_dir:
                    self._save_results_1([response], output_dir, [i])
        
        return responses
    def generate_rca_analysis_reasoner_coopetition_batch(self, alarm_data_list, output_dirs=None):
        """
        批量使用相同提示词模板，不同采样参数运行三次
        
        Args:
            alarm_data_list: 告警数据字典列表
            output_dirs: 输出目录路径列表（与alarm_data_list一一对应）
            
        Returns:
            merged_responses: 合并后的分析结果列表
        """
        print(f"批量生成Coopetition Reasoner分析，共{len(alarm_data_list)}个case...")
        
        # 定义三个不同的采样参数组合
        sampling_configs = [
            {"temperature": 0.1, "top_p": 0.9, "name": "Reasoner1"},
            {"temperature": 0.5, "top_p": 0.95, "name": "Reasoner2"},
            {"temperature": 0.8, "top_p": 1.0, "name": "Reasoner3"}
        ]
        
        # 为每个case准备存储
        all_responses_by_case = [[] for _ in range(len(alarm_data_list))]
        
        # 为每个采样配置批量处理
        for config_idx, config in enumerate(sampling_configs):
            reasoner_name = config['name']
            print(f"\n运行 {reasoner_name} 批次...")
            print(f"  参数: temperature={config['temperature']}, top_p={config['top_p']}")
            
            # 构建本批次所有case的提示词
            input_prompts = []
            for alarm_data in alarm_data_list:
                input_prompt = ROOT_CAUSE_Reasoner_Coopetiton.format(
                    alarm_type=alarm_data['alarm_type'],
                    semantic_labels=json.dumps(alarm_data['semantic_labels'], ensure_ascii=False, indent=2),
                    alarm_time=alarm_data['alarm_time'],
                    meta_output=alarm_data['meta_output'],
                    root_cause_candidates=json.dumps(alarm_data['root_cause_candidates'], ensure_ascii=False, indent=2)
                )
                input_prompts.append(input_prompt)
            
            # 批量调用大模型
            responses = vllm_invoke(
                self.llm,
                inputs=input_prompts,
                sampling_params=SamplingParams(
                    temperature=config['temperature'],
                    top_p=config['top_p'],
                    max_tokens=4096
                ),
                batch_size=16
            )
            
            # 将响应按case存储
            for i, response in enumerate(responses):
                if i < len(all_responses_by_case):
                    all_responses_by_case[i].append(response)
        
        # 合并每个case的所有响应
        merged_responses = []
        for case_responses in all_responses_by_case:
            merged_response = ""
            for i, resp in enumerate(case_responses):
                merged_response += f"=== {sampling_configs[i]['name']} ===\n{resp}\n\n{'='*50}\n\n"
            merged_responses.append(merged_response)
        
        # 保存结果
        if output_dirs and len(output_dirs) == len(merged_responses):
            for i, (response, output_dir) in enumerate(zip(merged_responses, output_dirs)):
                if output_dir:
                    self._save_results_1([response], output_dir, [i])
        
        return merged_responses

    def generate_rca_analysis_verifier_coopetiton_batch(self, alarm_data_list, output_dirs=None):
        """
        批量生成Verifier分析
        
        Args:
            alarm_data_list: 告警数据字典列表
            output_dirs: 输出目录路径列表（与alarm_data_list一一对应）
            
        Returns:
            responses: 分析结果列表
        """
        print(f"批量生成Coopetition Verifier分析，共{len(alarm_data_list)}个case...")
        
        # 构建所有输入提示词
        input_prompts = []
        for alarm_data in alarm_data_list:
            input_prompt = ROOT_CAUSE_Verifier_Coopetiton.format(
                alarm_type=alarm_data['alarm_type'],
                semantic_labels=json.dumps(alarm_data['semantic_labels'], ensure_ascii=False, indent=2),
                meta_output=alarm_data['meta_output'],
                root_cause_candidates=json.dumps(alarm_data['root_cause_candidates'], ensure_ascii=False, indent=2),
                reasoner_outputs=alarm_data['reasoner_outputs']
            )
            input_prompts.append(input_prompt)
        
        # 批量生成Verifier分析
        responses = vllm_invoke(
            self.llm,
            inputs=input_prompts,
            sampling_params=SamplingParams(temperature=0.5, top_p=0.9, max_tokens=4096),
            batch_size=16,
            prompt_output_paths=(
                [os.path.join(output_dir, "prompt.txt") for output_dir in output_dirs]
                if output_dirs else None
            ),
        )
        
        # 保存结果
        if output_dirs and len(output_dirs) == len(responses):
            for i, (response, output_dir) in enumerate(zip(responses, output_dirs)):
                if output_dir:
                    self._save_results_1([response], output_dir, [i])
        
        return responses
    def _save_results_meta(self,responses,output_dir,batch_indices):
        """保存批量结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, response in enumerate(responses):
            index = batch_indices[i]
            case_folder = f"{index}"
            case_dir = os.path.join(output_dir, case_folder)
            os.makedirs(case_dir, exist_ok=True)
            # 保存响应
            result_file = os.path.join(case_dir, f"raw_responses{index}.txt")
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Response {i+1} ===\n")
                f.write(response)
                f.write("\n\n" + "="*50 + "\n\n")
    def _save_results_1(self,responses,output_dir,batch_indices):
        """保存批量结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, response in enumerate(responses):
            # index = batch_indices[i]
            # case_folder = f"{index}"
            # case_dir = os.path.join(output_dir, case_folder)
            # os.makedirs(case_dir, exist_ok=True)
            # 保存响应
            result_file = os.path.join(output_dir, f"raw_responses.txt")
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Response {i+1} ===\n")
                f.write(response)
                f.write("\n\n" + "="*50 + "\n\n")
    def _save_results(self, responses, output_dirs, batch_indices):
        """
        批量保存结果
        
        Args:
            responses: 响应列表
            output_dirs: 输出目录列表（每个case一个目录）
            batch_indices: 批次索引列表（对应每个case的实际序号）
        """
        if len(responses) != len(output_dirs):
            print(f"警告: 响应数量 ({len(responses)}) 与输出目录数量 ({len(output_dirs)}) 不匹配")
            return
        
        if len(responses) != len(batch_indices):
            print(f"警告: 响应数量 ({len(responses)}) 与索引数量 ({len(batch_indices)}) 不匹配")
            return
        
        for i, (response, output_dir) in enumerate(zip(responses, output_dirs)):
            idx = batch_indices[i]
            
            # 确保目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存响应到 raw_responses.txt
            result_file = os.path.join(output_dir, "raw_responses.txt")
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Response for case {idx} ===\n")
                f.write(response)
                f.write("\n\n" + "="*50 + "\n\n")
            
            print(f"  保存 {idx} 号结果到: {result_file}")
