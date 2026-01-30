---
name: molgen-patterns
description: Patterns for molecular generation models including VAE, autoregressive, diffusion, GAN, RL-based, and GNN approaches.
---

# Molecular Generation Patterns

Patterns for implementing various molecular generation architectures.

## VAE-Based Generation

### VAE Architecture

```python
class MoleculeVAE(nn.Module):
    """Variational Autoencoder for molecules."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

        # Latent space projection
        self.fc_mu = nn.Linear(encoder.output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder.output_dim, latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """Compute VAE loss (reconstruction + KL divergence)."""
        # Reconstruction loss
        recon_loss = F.cross_entropy(recon.view(-1, recon.size(-1)), x.view(-1))

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + beta * kl_loss

        return total_loss, {"recon_loss": recon_loss, "kl_loss": kl_loss}

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from prior."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
```

### Beta-VAE Schedule

```python
def get_beta_schedule(
    epoch: int,
    total_epochs: int,
    beta_start: float = 0.0,
    beta_end: float = 1.0,
    warmup_epochs: int = 10,
) -> float:
    """Annealing schedule for KL weight."""
    if epoch < warmup_epochs:
        return beta_start + (beta_end - beta_start) * epoch / warmup_epochs
    return beta_end
```

## Autoregressive Generation

### Transformer Decoder

```python
class AutoregressiveDecoder(nn.Module):
    """Transformer-based autoregressive decoder for SMILES."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        max_length: int = 200,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Embedding(max_length, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output = nn.Linear(hidden_dim, vocab_size)
        self.max_length = max_length

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with teacher forcing."""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)

        # Embeddings
        h = self.embedding(x) + self.pos_encoding(positions)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        # Decode
        if memory is not None:
            h = self.transformer(h, memory, tgt_mask=mask)
        else:
            # Self-attention only (unconditional)
            h = self.transformer(h, h, tgt_mask=mask)

        return self.output(h)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        start_token: int,
        end_token: int,
        temperature: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()

        # Start with start token
        sequences = torch.full((batch_size, 1), start_token, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(self.max_length - 1):
            logits = self(sequences)[:, -1, :]
            logits = logits / temperature

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            sequences = torch.cat([sequences, next_token], dim=1)

            # Check for end token
            finished = finished | (next_token.squeeze(-1) == end_token)
            if finished.all():
                break

        return sequences
```

## Diffusion Models

### Denoising Diffusion

```python
class MoleculeDiffusion(nn.Module):
    """Diffusion model for molecules."""

    def __init__(
        self,
        denoiser: nn.Module,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps

        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to data."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return xt, noise

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Compute denoising loss."""
        batch_size = x0.size(0)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x0.device)

        noise = torch.randn_like(x0)
        xt, _ = self.forward_diffusion(x0, t, noise)

        predicted_noise = self.denoiser(xt, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample from model using DDPM."""
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            predicted_noise = self.denoiser(x, t_batch)

            alpha = 1 - self.betas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            alpha_cumprod_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)

            # DDPM update
            x = (1 / torch.sqrt(alpha)) * (
                x - (self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t]) * predicted_noise
            )

            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.betas[t])
                x = x + sigma * noise

        return x
```

## RL-Based Optimization

### REINFORCE for Molecule Optimization

```python
class MoleculeRLAgent(nn.Module):
    """RL agent for molecule optimization."""

    def __init__(
        self,
        generator: nn.Module,
        reward_fn: Callable[[list[str]], list[float]],
    ):
        super().__init__()
        self.generator = generator
        self.reward_fn = reward_fn

    def compute_policy_loss(
        self,
        smiles_list: list[str],
        log_probs: torch.Tensor,
        baseline: float = 0.0,
    ) -> torch.Tensor:
        """REINFORCE policy gradient loss."""
        rewards = torch.tensor(self.reward_fn(smiles_list), device=log_probs.device)

        # Advantage
        advantages = rewards - baseline

        # Policy gradient
        loss = -(log_probs * advantages).mean()

        return loss, rewards.mean()

    def train_step(
        self,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """Single RL training step."""
        # Generate molecules
        sequences, log_probs = self.generator.generate_with_log_probs(batch_size)

        # Decode to SMILES
        smiles_list = self.decode_sequences(sequences)

        # Compute loss
        loss, mean_reward = self.compute_policy_loss(smiles_list, log_probs)

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item(), "mean_reward": mean_reward.item()}
```

## GNN-Based Generation

### Message Passing for Molecular Graphs

```python
class MPNNLayer(nn.Module):
    """Message Passing Neural Network layer."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.message_fn = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_fn = nn.GRU(hidden_dim, node_dim)

    def forward(
        self,
        x: torch.Tensor,           # Node features (N, node_dim)
        edge_index: torch.Tensor,  # Edge indices (2, E)
        edge_attr: torch.Tensor,   # Edge features (E, edge_dim)
    ) -> torch.Tensor:
        row, col = edge_index

        # Compute messages
        messages = self.message_fn(
            torch.cat([x[row], x[col], edge_attr], dim=-1)
        )

        # Aggregate messages
        aggregated = torch.zeros_like(x[:, :messages.size(-1)])
        aggregated.scatter_add_(0, col.unsqueeze(-1).expand_as(messages), messages)

        # Update node features
        x_updated, _ = self.update_fn(aggregated.unsqueeze(0), x.unsqueeze(0))

        return x_updated.squeeze(0)

class MoleculeGNN(nn.Module):
    """GNN for molecular property prediction or generation."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MPNNLayer(node_dim if i == 0 else hidden_dim, edge_dim, hidden_dim)
            for i in range(num_layers)
        ])

        self.pool = GlobalAttentionPool(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)

        # Global pooling
        graph_embedding = self.pool(x, batch)

        return self.output(graph_embedding)
```

## Generation Quality Metrics

```python
def evaluate_generation(
    generated_smiles: list[str],
    reference_smiles: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate generated molecules."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    # Validity
    valid_mols = [Chem.MolFromSmiles(s) for s in generated_smiles]
    valid_mols = [m for m in valid_mols if m is not None]
    validity = len(valid_mols) / len(generated_smiles)

    # Uniqueness
    unique_smiles = set(Chem.MolToSmiles(m, canonical=True) for m in valid_mols)
    uniqueness = len(unique_smiles) / len(valid_mols) if valid_mols else 0

    # Internal diversity
    if len(valid_mols) > 1:
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps = [fpgen.GetFingerprint(m) for m in valid_mols]
        similarities = []
        for i, fp1 in enumerate(fps):
            for fp2 in fps[i+1:]:
                similarities.append(DataStructs.TanimotoSimilarity(fp1, fp2))
        diversity = 1 - np.mean(similarities)
    else:
        diversity = 0

    metrics = {
        "validity": validity,
        "uniqueness": uniqueness,
        "diversity": diversity,
    }

    # Novelty (if reference provided)
    if reference_smiles:
        reference_set = set(reference_smiles)
        novel = sum(1 for s in unique_smiles if s not in reference_set)
        metrics["novelty"] = novel / len(unique_smiles) if unique_smiles else 0

    return metrics
```
