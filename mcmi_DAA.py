import numpy as np
import matplotlib.pyplot as plt
import time
import tqdm
from typing import Callable, Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import heapq
from datetime import timezone

@dataclass
class ExperimentConfig:
    width: int = 8
    height: int = 8
    obstacles: List[Tuple[int, int]] = None
    goal_states: List[Tuple[int, int]] = None
    goal_rewards: List[float] = None
    step_penalty: float = 0.01
    gamma: float = 0.95
    learning_rate: float = 0.1
    epsilon: float = 0.1
    decay_rate: float = 0.995
    n_steps: int = 100000
    priority_threshold: float = 0.01
    wind_prob: float = 0.1
    slip_prob: float = 0.1
    save_results: bool = True
    output_dir: str = "results"


class PrioritizedMCMI: # Agent sử dụng MCMI với ưu tiên cập nhật
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.gamma = config.gamma
        self.learning_rate = config.learning_rate
        self.epsilon = config.epsilon
        self.decay_rate = config.decay_rate
        self.priority_threshold = config.priority_threshold
        
        self.state_values = None
        self.state_counts = None
        self.priority_queue = []
        self.predecessors = {}
        
        self.metrics = {
            'convergence_history': [],
            'episode_lengths': [],
            'total_rewards': [],
            'value_updates': [],
            'exploration_ratio': [],
            'computation_time': 0
        }

    def reset(self, n_states: int): # Đặt lại trạng thái của agent
        """Reset agent's state"""
        self.state_values = np.zeros(n_states)
        self.state_counts = np.zeros(n_states)
        self.priority_queue = []
        self.predecessors = {s: set() for s in range(n_states)}
        self.metrics = {k: [] for k in self.metrics.keys()}
        self.metrics['computation_time'] = 0


    def _get_action(self, state: int, env: Any, policy: Optional[Callable] = None) -> int: # Lấy hành động dựa trên chiến lược epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(env.n_actions)
        if policy is None:
            # Chính sách mặc định dựa trên hàm giá trị
            possible_actions = range(env.n_actions)
            action_values = []
            for action in possible_actions:
                next_probs = env.P[state, action]
                value = np.sum(next_probs * self.state_values)
                action_values.append(value)
            return np.argmax(action_values)
        return policy(state)


    def _process_trajectory(self, trajectory: list, rewards: list) -> float: # Xử lý toàn bộ trajectory để cập nhật giá trị trạng thái
        max_value_change = 0
        for t, state in enumerate(trajectory):
            G = 0
            weight = 1.0
            for i, r in enumerate(rewards[t:]):
                G += weight * r * (self.gamma ** i)
                weight *= self.decay_rate
            old_value = self.state_values[state]
            self.state_counts[state] += 1
            lr = self.learning_rate / np.sqrt(self.state_counts[state])
            new_value = old_value + lr * (G - old_value)
            self.state_values[state] = new_value
            
            # Ưu tiên cập nhật
            value_change = abs(new_value - old_value)
            max_value_change = max(max_value_change, value_change)
            if value_change > self.priority_threshold:
                heapq.heappush(self.priority_queue, (-value_change, state))
                
        return max_value_change


    def _process_priority_updates(self, env: Any, max_updates: int = 10): # Xử lý cập nhật ưu tiên từ hàng đợi
        updates = 0
        while self.priority_queue and updates < max_updates:
            _, state = heapq.heappop(self.priority_queue)
            old_value = self.state_values[state]
            
            # Tính giá trị mới cho trạng thái dựa trên các hành động
            # và xác suất chuyển tiếp
            new_value = 0
            for action in range(env.n_actions):
                next_probs = env.P[state, action]
                reward = env.R[state]
                expected_value = np.sum(next_probs * (
                    reward + self.gamma * self.state_values
                ))
                new_value = max(new_value, expected_value)
            
            # Cập nhật giá trị trạng thái nếu thay đổi lớn hơn ngưỡng ưu tiên

            value_change = abs(new_value - old_value) # Tính  giá trị thay đổi
            if value_change > self.priority_threshold:
                self.state_values[state] = new_value
                for pred in self.predecessors[state]:
                    heapq.heappush(self.priority_queue, 
                                 (-value_change * self.gamma, pred))
            
            updates += 1



    def estimate_values(self, env: Any, early_stop_threshold: float = 0.001) -> np.ndarray:  # Ước lượng giá trị trạng thái với MCMI và sweeping ưu tiên
        if self.state_values is None: # Kiểm tra nếu giá trị trạng thái chưa được khởi tạo
            self.reset(env.n_states)
            
        start_time = time.time()
        steps_taken = 0
        episode = 0
        window_size = 100
        recent_changes = []
        with tqdm.tqdm(total=self.config.n_steps, 
                      desc=f"MCMI Learning (γ={self.gamma})") as pbar:
            while steps_taken < self.config.n_steps:
                episode += 1
                state = env.reset()
                trajectory = []
                rewards = []
                done = False
                episode_steps = 0
    
                while not done and steps_taken < self.config.n_steps:
                    trajectory.append(state)
                    action = self._get_action(state, env)
                    next_state, reward, done, _ = env.step(action)
                    rewards.append(reward)
        
                    self.predecessors[next_state].add(state)
                    
                    state = next_state
                    steps_taken += 1
                    episode_steps += 1
                    pbar.update(1)
                    
                    if np.random.random() > self.gamma:
                        done = True
                value_change = self._process_trajectory(trajectory, rewards)
                self._process_priority_updates(env)
      
                self.metrics['episode_lengths'].append(episode_steps)
                self.metrics['total_rewards'].append(sum(rewards))
                self.metrics['value_updates'].append(value_change)
                self.metrics['exploration_ratio'].append(
                    len(set(trajectory)) / env.n_states
                )
                
                if episode % 10 == 0:
                    self.metrics['convergence_history'].append(
                        (steps_taken, np.mean(self.metrics['value_updates'][-10:]))
                    )
                
                # Kiểm tra early stopping
                recent_changes.append(value_change)
                if len(recent_changes) >= window_size:
                    avg_change = np.mean(recent_changes[-window_size:])
                    if avg_change < early_stop_threshold:
                        print(f"\nEarly stopping at step {steps_taken}")
                        break
                
                # Cập nhật tỷ lệ epsilon
                self.epsilon *= self.decay_rate
        
        self.metrics['computation_time'] = time.time() - start_time
        return self.state_values



    def analyze_performance(self) -> Dict: # Phân tích hiệu suất của quá trình học
        return {
            'average_return': np.mean(self.metrics['total_rewards']),
            'std_return': np.std(self.metrics['total_rewards']),
            'success_rate': len([r for r in self.metrics['total_rewards'] if r > 0]) / 
                          len(self.metrics['total_rewards']),
            'average_steps': np.mean(self.metrics['episode_lengths']),
            'final_convergence': self.metrics['convergence_history'][-1][1]
            if self.metrics['convergence_history'] else None,
            'exploration_coverage': np.mean(self.metrics['exploration_ratio']),
            'computation_time': self.metrics['computation_time']
        }



    def visualize_learning_process(self, save_path: Optional[str] = None): # Hiển thị quá trình học với các biểu đồ phân tích
        fig = plt.figure(figsize=(18, 12))
        gs = plt.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
        fig.suptitle('MCMI Learning Analysis', y=0.99, fontsize=16)
        
        # Lịch sử hội tụ
        ax1 = fig.add_subplot(gs[0, 0])
        if self.metrics['convergence_history']:
            steps, errors = zip(*self.metrics['convergence_history'])
            ax1.plot(steps, errors)
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Average Value Change')
            ax1.set_title('Convergence History', pad=20)
            ax1.grid(True)
        
        # Chiều dài Episode
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.metrics['episode_lengths'])
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Lengths', pad=20)
        ax2.grid(True)
        
        # Tổng phần thưởng mỗi episode
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.metrics['total_rewards'])
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Total Reward')
        ax3.set_title('Episode Rewards', pad=20)
        ax3.grid(True)
        
        # Phân phối cập nhật giá trị
        ax4 = fig.add_subplot(gs[1, 0])
        if self.metrics['value_updates']:
            ax4.hist(self.metrics['value_updates'], bins=30, edgecolor='black')
            ax4.set_xlabel('Value Change')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Value Update Distribution', pad=20)
            ax4.grid(True)
        
        # Tỷ lệ khám phá không gian trạng thái
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(self.metrics['exploration_ratio'])
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Exploration Ratio')
        ax5.set_title('State Space Coverage', pad=20)
        ax5.grid(True)
        
        # Phân phối giá trị trạng thái
        ax6 = fig.add_subplot(gs[1, 2])
        if self.state_values is not None:
            ax6.hist(self.state_values, bins=30, edgecolor='black')
            ax6.set_xlabel('State Value')
            ax6.set_ylabel('Frequency')
            ax6.set_title('State Value Distribution', pad=20)
            ax6.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()



class EnhancedGridWorld: # Môi trường lưới với các tính năng mở rộng như gió, trượt và phần thưởng gradient
    def __init__(self, config: ExperimentConfig):
        self.width = config.width
        self.height = config.height
        self.n_states = self.width * self.height
        self.n_actions = 4
        self.state = 0
        
        self.wind_prob = config.wind_prob
        self.slip_prob = config.slip_prob
        
        self.obstacles = ([y * self.width + x for x, y in config.obstacles] 
                         if config.obstacles else [])
        self.goal_states = ([y * self.width + x for x, y in config.goal_states]
                           if config.goal_states else [self.width * self.height - 1])
        self.goal_rewards = ({state: reward for state, reward in 
                            zip(self.goal_states, config.goal_rewards)}
                           if config.goal_rewards 
                           else {s: 1.0 for s in self.goal_states})
        
        self.step_penalty = config.step_penalty 
        self._create_transition_matrix()
        self._create_reward_vector()
        


    def _create_transition_matrix(self): # Tạo ma trận chuyển tiếp với các hiệu ứng gió và trượt
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            if s in self.obstacles or s in self.goal_states:
                self.P[s, :, s] = 1.0
                continue
            x, y = s % self.width, s // self.width
            for action in range(self.n_actions):
                # Tính toán trạng thái tiếp theo dựa trên hành động và các hiệu ứng gió, trượt
                next_x, next_y = x, y
                if action == 0:  # Up
                    next_y = max(0, y-1)
                elif action == 1:  # Right
                    next_x = min(self.width-1, x+1)
                elif action == 2:  # Down
                    next_y = min(self.height-1, y+1)
                else:  # Left
                    next_x = max(0, x-1)
                
                next_s = next_y * self.width + next_x
                if next_s in self.obstacles:
                    next_s = s
                # Xác suất chuyển tiếp chính
                main_prob = (1 - self.wind_prob) * (1 - self.slip_prob)
                self.P[s, action, next_s] += main_prob
                # Xác suất chuyển tiếp với gió, Gió có thể làm lệch hướng di chuyển
                for wind_direction in [-1, 1]:  
                    wind_x = x
                    wind_y = y
                    if action in [0, 2]:  
                        wind_x = max(0, min(self.width-1, x + wind_direction))
                    else:  
                        wind_y = max(0, min(self.height-1, y + wind_direction))
                    wind_s = wind_y * self.width + wind_x
                    if wind_s in self.obstacles:
                        wind_s = s
                    self.P[s, action, wind_s] += self.wind_prob / 2
                # Xác suất chuyển tiếp với trượt
                self.P[s, action, s] += self.slip_prob
            # Chuẩn hóa xác suất chuyển tiếp
            for action in range(self.n_actions):
                self.P[s, action] /= np.sum(self.P[s, action])


    def _create_reward_vector(self): # Tạo vector phần thưởng với phần thưởng gradient gần mục tiêu
        self.R = np.zeros(self.n_states) - self.step_penalty  
        # Thêm phần thưởng cho các trạng thái mục tiêu
        for goal_state, reward in self.goal_rewards.items():
            self.R[goal_state] = reward
            # Thêm gradient reward gần mục tiêu
            x_goal, y_goal = goal_state % self.width, goal_state // self.width
            for x in range(self.width):
                for y in range(self.height):
                    state = y * self.width + x
                    if state not in self.obstacles and state not in self.goal_states:
                        distance = abs(x - x_goal) + abs(y - y_goal)
                        self.R[state] += reward * 0.1 / (distance + 1)
      
        for obstacle in self.obstacles: # Đặt phần thưởng cho các trạng thái chướng ngại vật
            self.R[obstacle] = 0


    def step(self, action: int) -> Tuple[int, float, bool, Dict]: # Thực hiện hành động và trả về trạng thái tiếp theo, phần thưởng, trạng thái kết thúc
        if action not in range(self.n_actions):
            raise ValueError(f"Invalid action {action}")   
        # Xác định trạng thái tiếp theo dựa trên ma trận chuyển tiếp
        next_state = np.random.choice(self.n_states, p=self.P[self.state, action])
        reward = self.R[next_state]
        done = next_state in self.goal_states
        self.state = next_state
        return next_state, reward, done, {}  
    

    def reset(self) -> int: # Đặt lại môi trường về trạng thái ban đầu
        valid_states = [s for s in range(self.n_states) 
                       if s not in self.obstacles and s not in self.goal_states]
        self.state = np.random.choice(valid_states)
        return self.state
    

    def visualize_state(self, state_values: np.ndarray, policy: Optional[Callable] = None, 
                       title: str = "Grid World State Values", save_path: Optional[str] = None): # Hiển thị giá trị trạng thái và chính sách

        plt.figure(figsize=(12, 8))
        # Hiển thị giá trị trạng thái dưới dạng lưới
        grid_values = state_values.reshape(self.height, self.width)
        plt.imshow(grid_values, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='State Value')
        # Thêm lưới và nhãn
        if policy:
            for y in range(self.height):
                for x in range(self.width):
                    state = y * self.width + x
                    if state not in self.obstacles and state not in self.goal_states:
                        action = policy(state)
                        dx = dy = 0
                        if action == 0: dy = -0.3   # Up
                        elif action == 1: dx = 0.3  # Right
                        elif action == 2: dy = 0.3  # Down
                        elif action == 3: dx = -0.3 # Left
                        
                        plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1,
                                fc='white', ec='white', alpha=0.7, 
                                length_includes_head=True)
        # Hiển thị chướng ngại vật và mục tiêu
        for obs in self.obstacles:
            x, y = obs % self.width, obs // self.width
            plt.plot(x, y, 'rx', markersize=15, markeredgewidth=2, 
                    label='Obstacle' if obs == self.obstacles[0] else "")
        for goal in self.goal_states:
            x, y = goal % self.width, goal // self.width
            plt.plot(x, y, 'g*', markersize=15, markeredgewidth=2,
                    label='Goal' if goal == self.goal_states[0] else "")
        
        # Thiết lập tiêu đề và nhãn
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.title(title, pad=20)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        # Thêm chú thích
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), 
                  loc='center left', bbox_to_anchor=(1.3, 1))
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()



def run_enhanced_experiment(config: ExperimentConfig): # Chạy thử nghiệm với môi trường lưới mở rộng và agent MCMI ưu tiên
    env = EnhancedGridWorld(config) # Khởi tạo môi trường lưới mở rộng
    agent = PrioritizedMCMI(config) # Khởi tạo agent MCMI ưu tiên
    print("\nStarting MCMI learning with enhanced features...")
    print(f"Environment size: {config.width}x{config.height}")
    print(f"Number of obstacles: {len(config.obstacles) if config.obstacles else 0}")
    print(f"Number of goals: {len(config.goal_states) if config.goal_states else 1}")
    
    values = agent.estimate_values(env) # Ước lượng giá trị trạng thái
    
    metrics = agent.analyze_performance() # Phân tích hiệu suất
    print("\nLearning Results:")
    for key, value in metrics.items(): 
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    env.visualize_state(values) # Hiển thị giá trị trạng thái
    agent.visualize_learning_process() # Hiển thị quá trình học
    return values, metrics


def run_experiments_with_multiple_gamma(base_config: ExperimentConfig, gamma_values: List[float]): # Chạy thử nghiệm với nhiều giá trị gamma khác nhau
    all_results = []
    
    for gamma in gamma_values: 
        print(f"\n{'='*50}")
        print(f"Running experiment with gamma = {gamma}")
        print(f"{'='*50}")
        
        config = ExperimentConfig(**base_config.__dict__) # Tạo bản sao cấu hình cơ bản
        config.gamma = gamma # Cập nhật giá trị gamma
        
        values, metrics = run_enhanced_experiment(config) # Chạy thử nghiệm với cấu hình đã cập nhật
        

        all_results.append({ 
            'gamma': gamma,
            'values': values,
            'metrics': metrics
        })
    
    plot_gamma_comparison(all_results)
    print_comparison_table(all_results)


def plot_gamma_comparison(results: List[Dict]): # Vẽ biểu đồ so sánh hiệu suất học với các giá trị gamma khác nhau
    fig = plt.figure(figsize=(20, 12))
    metrics_to_plot = [
        ('average_return', 'Average Return per Episode', 'Return Value'),
        ('success_rate', 'Goal Reaching Success Rate', 'Success Rate (%)'),
        ('average_steps', 'Average Steps to Goal', 'Number of Steps'),
        ('final_convergence', 'Value Function Convergence', 'Convergence Error'),
        ('exploration_coverage', 'State Space Exploration', 'Coverage Rate (%)'),
        ('computation_time', 'Algorithm Runtime', 'Time (seconds)')
    ]
    
    for idx, (metric_key, title, ylabel) in enumerate(metrics_to_plot, 1): # Tạo các biểu đồ cho từng metric
        ax = fig.add_subplot(2, 3, idx)
        
        x = [r['gamma'] for r in results]
        y = [r['metrics'][metric_key] for r in results]
        
        ax.plot(x, y, 'o-', linewidth=2, markersize=8, color='#2E86C1')
     
        ax.set_title(title, pad=10, fontsize=11)
        ax.set_xlabel('Discount Factor (γ)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        
    
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(labelsize=9)
   
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('#666666')
            
        # Format y-axis để tránh chồng chéo
        if metric_key in ['success_rate', 'exploration_coverage']:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    

    plt.suptitle('Analysis of Discount Factor (γ) Impact on Learning Performance', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.35)
    
    plt.show()

def print_comparison_table(results: List[Dict]): # In bảng so sánh các metric học tập với các giá trị gamma khác nhau
    print("\nComparison of Learning Metrics Across Different Discount Factors:")
    print("-" * 120)
    
    headers = [
        "Gamma (γ)", 
        "Return/Episode", 
        "Success Rate", 
        "Steps/Goal", 
        "Convergence", 
        "Coverage %", 
        "Runtime (s)"
    ]
    
    print(f"{headers[0]:^10} {headers[1]:^15} {headers[2]:^15} {headers[3]:^15} "
          f"{headers[4]:^15} {headers[5]:^15} {headers[6]:^15}")
    print("-" * 120)
    
    for result in results:
        gamma = result['gamma']
        metrics = result['metrics']
        print(f"{gamma:^10.3f} {metrics['average_return']:^15.3f} "
              f"{metrics['success_rate']:^15.3%} {metrics['average_steps']:^15.1f} "
              f"{metrics['final_convergence']:^15.5f} "
              f"{metrics['exploration_coverage']:^15.1%} "
              f"{metrics['computation_time']:^15.2f}")
    print("-" * 120)

if __name__ == "__main__": # có thể thay đổi cấu hình thử nghiệm 
    base_config = ExperimentConfig(
        width=24,
        height=24,
        obstacles=[
                (12, 5), (11, 6), (13, 6), (10, 7), (14, 7), #tùy ý tạo 1 chướng ngại vật bất kỳ
                (9, 8), (15, 8), (8, 9), (16, 9),
                (7, 10), (17, 10), (6, 11), (18, 11),
                (6, 12), (18, 12), (7, 13), (17, 13),
                (8, 14), (16, 14), (9, 15), (15, 15),
                (10, 16), (14, 16), (11, 17), (13, 17), (12, 18)
            ],       
        goal_states=[(23,23), (0,23)], # có thể thêm mục tiêu
        goal_rewards=[10, 5], # phần thưởng cho các mục tiêu
        step_penalty=0.03,  # phần thưởng bước
        learning_rate=0.5, # tốc độ học
        epsilon=0.3, # xác suất khám phá
        decay_rate=0.995, # tỷ lệ giảm epsilon
        n_steps=100000,  # số bước học
        priority_threshold=0.01, # ngưỡng ưu tiên cập nhật
        wind_prob=0.4, # xác suất gió, có thể thay đổi
        slip_prob=0.2, # xác suất trượt, có thể thay đổi
    )
    gamma_values = [0.99]
    run_experiments_with_multiple_gamma(base_config, gamma_values)