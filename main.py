import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import networkx as nx

torch.set_float32_matmul_precision('medium')
malicious_percent = 0.2
malicious_types = ["label_flipping", "random_update", "byzantine", "free_riding"]

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        MNIST(root="data", train=True, download=True)
        MNIST(root="data", train=False, download=True)

    def setup(self, stage=None):
        self.train = MNIST(root="data", train=True, transform=self.transform)
        self.test = MNIST(root="data", train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

class SimpleMNISTNet(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
class GossipNode:
    def __init__(self, model, dataloader, id, test_loader=None, node_type="benign"):
        self.model = model
        self.id = id
        self.dataloader = dataloader
        self.test_loader = test_loader
        self.node_type = node_type
        self.cached_weights = None  # For free-riding nodes

    def local_train(self):
        """Train locally unless the node is malicious in a way that changes training behavior."""

        # ---------- FREE RIDING ----------
        if self.node_type == "free_riding":
            # Do not train at all. Just keep weights as they were.
            if self.cached_weights is None:
                self.cached_weights = {k: v.detach().clone()
                                       for k, v in self.model.state_dict().items()}
            return

        # ---------- LABEL FLIPPING ----------
        if self.node_type == "label_flipping":
            # Wrap dataloader so labels are inverted (e.g., MNIST: y -> 9 - y)
            flipped_loader = self._make_label_flipped_loader()
            dl = flipped_loader
        else:
            dl = self.dataloader

        # ---------- Normal / Byzantine / Random-update training ----------
        trainer = pl.Trainer(
            max_epochs=3,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            accelerator="cuda",
            devices=1,
        )
        trainer.fit(self.model, dl)

        # Store weights post-training for free-riding consistency
        self.cached_weights = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }

    # -------------------------
    #   MALICIOUS WEIGHT RETURN
    # -------------------------

    def get_weights(self):
        """Return weights after applying node-specific malicious behavior."""

        # First get weights from model or cached copy
        if self.node_type == "free_riding":
            # Return stale weights (not trained this round)
            return {k: v.detach().clone() for k, v in self.cached_weights.items()}

        weights = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        # ---------- RANDOM UPDATE ----------
        if self.node_type == "random_update":
            for k in weights:
                weights[k] = torch.randn_like(weights[k])

        # ---------- BYZANTINE ----------
        if self.node_type == "byzantine":
            for k in weights:
                # Multiply by a large factor to blow up aggregation
                weights[k] = weights[k] * torch.randn(1).item() * 50.0

        # Benign + label-flipping are already handled (label flipping affects training only)

        return weights

    # -------------------------
    #   HELPERS
    # -------------------------

    def _make_label_flipped_loader(self):
        """Return a dataloader that flips labels such as y -> 9 - y for MNIST."""
        from torch.utils.data import DataLoader

        class FlippedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                x, y = self.dataset[idx]
                y = 9 - y  # MNIST label flip
                return x, y

        flipped_ds = FlippedDataset(self.dataloader.dataset)
        return DataLoader(flipped_ds, batch_size=self.dataloader.batch_size, shuffle=True)

    def set_weights(self, new_state):
        self.model.load_state_dict(new_state)
    
    def evaluate_accuracy(self):
        """Evaluate accuracy on the node's test_loader."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                logits = self.model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / total

    def gossip_with(self, other, alpha=0.5):
        w1 = self.get_weights()
        w2 = other.get_weights()

        mixed = {k: alpha * w1[k] + (1 - alpha) * w2[k] for k in w1}

        self.set_weights(mixed)

def make_nodes(num_nodes=5, batch_size=64):
    data = MNISTDataModule(batch_size=batch_size)
    data.prepare_data()
    data.setup()
    malicious_num = int(num_nodes * malicious_percent)
    malicious_ids = set()
    for _ in range(malicious_num):
        while True:
            mid = torch.randint(0, num_nodes, (1,)).item()
            if mid not in malicious_ids:
                malicious_ids.add(mid)
                break
    
    full_train = data.train
    N = len(full_train)
    chunk = N // num_nodes

    test_loader = data.val_dataloader()

    nodes = []
    for i in range(num_nodes):
        start = i * chunk
        end = (i + 1) * chunk if i < num_nodes - 1 else N  # last node takes remainder

        subset = torch.utils.data.Subset(full_train, range(start, end))
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        model = SimpleMNISTNet()
        node_type = "benign"
        if i in malicious_ids:
            node_type = malicious_types[torch.randint(0, len(malicious_types), (1,)).item()]
            print(f"Node {i} is malicious: {node_type}")
            # Here you can modify the model or training process based on the malicious type
            # For simplicity, we just print the type

        nodes.append(
            GossipNode(model, train_loader, id=i, test_loader=test_loader, node_type=node_type)
        )
    return nodes



def build_small_world_topology(nodes, k=4, p=0.1):
    """
    Builds a small-world graph (Watts–Strogatz) and returns neighbors for each node.
    k: each node is connected to k nearest neighbors
    p: probability of rewiring edges
    """
    num_nodes = len(nodes)
    G = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p)
    neighbors = {i: list(G.neighbors(i)) for i in range(num_nodes)}
    return neighbors

def run_gossip_simulation_sw(nodes, rounds=20, alpha=0.5, k=4, p=0.1):
    # Build small-world neighbor mapping once
    neighbors = build_small_world_topology(nodes, k=k, p=p)

    for r in range(rounds):
        print(f"\n--- Round {r+1} ---")

        # Each node trains locally
        for node in nodes:
            node.local_train()

        # Gossip step: each node gossips with all neighbors
        for node in nodes:
            for neighbor_idx in neighbors[node.id]:
                neighbor_node = nodes[neighbor_idx]
                node.gossip_with(neighbor_node, alpha=alpha)

        # Accuracy report
        for node in nodes:
            acc = node.evaluate_accuracy()
            print(f"Node {node.id} accuracy: {acc:.4f}")


if __name__ == "__main__":
    num_nodes = 10
    nodes = make_nodes(num_nodes=num_nodes, batch_size=64)
    run_gossip_simulation_sw(nodes, rounds=5, alpha=0.5, k=4, p=0.1)
