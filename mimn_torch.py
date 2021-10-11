import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MIMNCell(nn.Module):
    def __init__(self, controller_units, memory_size, memory_vector_dim, read_head_num, write_head_num, embedding, output_dim=None, clip_value=20, shift_range=1, batch_size=128, mem_induction=0, util_reg=False, sharp_value=2.,):
        super(MIMNCell, self).__init__()
        self.controller_units = controller_units
        self.embedding_dim = embedding
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.mem_induction = mem_induction
        self.util_reg = util_reg
        self.clip_value = clip_value
        self.sharp_value = sharp_value
        self.shift_range = shift_range
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learned_init_w = nn.Linear(1, memory_vector_dim, bias=False)
        self.controller = nn.GRUCell(self.embedding_dim + self.memory_vector_dim, self.controller_units)

        self.step = 0
        self.output_dim = output_dim
        self.num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1
        self.num_heads = self.read_head_num + self.write_head_num
        self.total_parameter_num = self.num_parameters_per_head * self.num_heads + self.memory_vector_dim * 2 * self.write_head_num
        self.controller_w = nn.Linear(self.controller_units, self.total_parameter_num)
        self.controller_w_reg = nn.Linear(self.controller_units + self.memory_size, self.total_parameter_num)

        self.M_t = nn.Parameter(torch.empty(self.memory_size, self.memory_vector_dim).uniform_(-1, 1)).type(torch.FloatTensor)
        self.key_M_t = nn.Parameter(torch.empty(self.memory_size, self.memory_vector_dim).uniform_(-1, 1)).type(torch.FloatTensor)
        self.controller_init_state_train = torch.randn(self.batch_size, self.controller_units).to(self.device)
        self.controller_init_state_test = torch.randn(1, self.controller_units).to(self.device)
        self.read_vector_list_one = nn.Parameter(torch.empty(self.memory_vector_dim).uniform_(-1, 1)).type(torch.FloatTensor)
        self.w_list_one = nn.Parameter(torch.empty(self.memory_vector_dim).uniform_(-1, 1)).type(torch.FloatTensor)

    def forward(self, x, prev_state):
        prev_read_vector_list = prev_state["read_vector_list"]

        controller_input = torch.cat([x] + prev_read_vector_list, dim=1)

        controller_output = self.controller(controller_input, prev_state["controller_state"])

        controller_state = controller_output


        if self.util_reg:
            max_q = 250.0
            prev_w_aggre = prev_state["w_aggre"] / max_q
            controller_par = torch.cat([controller_output, prev_w_aggre.detach()], dim=1)
            parameters = self.controller_w_reg(controller_par)
        else:
            controller_par = controller_output
            parameters = self.controller_w(controller_par)


        parameters = torch.clamp(parameters, -self.clip_value, self.clip_value)

        head_parameter_list = torch.chunk(parameters[:, :self.num_parameters_per_head * self.num_heads], self.num_heads, dim=1)
        erase_add_list = torch.chunk(parameters[:, self.num_parameters_per_head * self.num_heads:], 2 * self.write_head_num, dim=1)

        prev_M = prev_state["M"]
        key_M = prev_state["key_M"]
        w_list = []
        write_weight = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = nn.Tanh()(head_parameter[:, 0:self.memory_vector_dim])
            beta = (nn.Softplus()(head_parameter[:, self.memory_vector_dim]) + 1) * self.sharp_value
            w = self.addressing(k, beta, key_M, prev_M)
            if self.util_reg and i == 1:
                s = nn.Softmax()(head_parameter[:, self.memory_vector_dim + 2:self.memory_vector_dim + 2 + (self.shift_range * 2 + 1)])
                gamma = 2*(nn.Softplus()(head_parameter[:, -1]) + 1)*self.sharp_value
                w = self.capacity_overflow(w, s, gamma)
                write_weight.append(self.capacity_overflow(w.detach(), s, gamma))

            w_list.append(w)

        read_w_list = w_list[:self.read_head_num]
        read_vector_list = []
        for i in range(self.read_head_num):
            read_vector = torch.sum(torch.unsqueeze(read_w_list[i], dim=2) * prev_M, dim=1)
            read_vector_list.append(read_vector)

        write_w_list = w_list[self.read_head_num:]


        M = prev_M
        sum_aggre = prev_state["sum_aggre"]

        for i in range(self.write_head_num):
            w = torch.unsqueeze(write_w_list[i], dim=2)
            erase_vector = torch.unsqueeze(nn.Sigmoid()(erase_add_list[i * 2]), dim=1)
            add_vector = torch.unsqueeze(nn.Tanh()(erase_add_list[i * 2 + 1]), dim=1)
            M = M * (torch.ones(M.size()).to(self.device) - torch.matmul(w, erase_vector)) + torch.matmul(w, add_vector)
            sum_aggre += torch.matmul(w.detach(), add_vector)

        w_aggre = prev_state["w_aggre"]
        if self.util_reg:
            for ww in write_weight:
                w_aggre += ww
        else:
            for ww in write_w_list:
                w_aggre += ww

        self.step += 1
        return {
            "controller_state": controller_state,
            "read_vector_list": read_vector_list,
            "w_list": w_list,
            "M": M,
            "key_M": key_M,
            "w_aggre": w_aggre,
            "sum_aggre": sum_aggre
        }

    def expand(self, x, dim, N):
        return torch.cat([torch.unsqueeze(x, dim) for _ in range(N)], dim=dim)

    def learned_init(self, units):
        x = torch.ones(1, 1).to(self.device)
        x = self.learned_init_w(x)
        return torch.squeeze(x)

    def addressing(self, k, beta, key_M, prev_M):
        # Cosine Similarity
        def cosine_similarity(key, M):
            key = torch.unsqueeze(key, dim=2)
            inner_product = torch.matmul(M, key)
            k_norm = torch.sqrt(torch.sum(torch.square(key), dim=1, keepdim=True))
            M_norm = torch.sqrt(torch.sum(torch.square(M), dim=2, keepdim=True))
            norm_product = M_norm * k_norm
            K = torch.squeeze(inner_product / (norm_product + 1e-8))
            return K

        K = 0.5*(cosine_similarity(k,key_M) + cosine_similarity(k,prev_M))
        K_amplified = torch.exp(torch.unsqueeze(beta, dim=1) * K)
        w_c = K_amplified / torch.sum(K_amplified, dim=1, keepdim=True)

        return w_c

    def capacity_overflow(self, w_g, s, gamma):
        s = torch.cat([s[:, :self.shift_range + 1],
                       torch.zeros([s.shape[0], self.memory_size - (self.shift_range * 2 + 1)]).to(self.device),
                       s[:, -self.shift_range:]], dim=1)
        t = torch.cat([torch.flip(s, dims=[1]), torch.flip(s, dims=[1])], dim=1)
        s_matrix = torch.stack(
            [t[:, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)],
            dim=1
        )
        w_ = torch.sum(torch.unsqueeze(w_g, dim=1) * s_matrix, dim=2)
        w_sharpen = torch.pow(w_, torch.unsqueeze(gamma, dim=1))
        w = w_sharpen / torch.sum(w_sharpen, dim=1, keepdim=True)

        return w

    def capacity_loss(self, w_aggre):
        loss = 0.001 * torch.mean((w_aggre - torch.mean(w_aggre, dim=-1, keepdim=True))**2 / self.memory_size / self.batch_size)
        return loss

    def zero_state(self, batch_size):
        read_vector_list = [self.expand(nn.Tanh()(self.read_vector_list_one), dim=0, N=batch_size)
                            for i in range(self.read_head_num)]

        w_list = [self.expand(nn.Softmax()(self.w_list_one), dim=0, N=batch_size)
                  for i in range(self.read_head_num + self.write_head_num)]

        controller_init_state = self.controller_init_state_train

        if batch_size == 1:
            controller_init_state = self.controller_init_state_test

        # M_t = torch.randn(self.memory_size, self.memory_vector_dim).to(self.device)

        M = self.expand(nn.Tanh()(self.M_t), dim=0, N=batch_size)

        # key_M_t = torch.randn(self.memory_size, self.memory_vector_dim).to(self.device)
        key_M = self.expand(nn.Tanh()(self.key_M_t), dim=0, N=batch_size)

        sum_aggre = torch.zeros([batch_size, self.memory_size, self.memory_vector_dim]).to(self.device)
        zero_weight_vector = torch.zeros([batch_size, self.memory_size]).to(self.device)

        state = {
            "controller_state": controller_init_state,
            "read_vector_list": read_vector_list,
            "w_list": w_list,
            "M": M,
            "w_aggre": zero_weight_vector,
            "key_M": key_M,
            "sum_aggre": sum_aggre

        }
        return state

