# Glossary

`CodeLeWM`: A latent world model for Python code edit transitions.

`CodeState`: A deterministic context capsule for a changed Python symbol or
region. It contains primary code plus bounded local context.

`EditAction`: The conditioning signal for a transition. It can be natural text,
an abstract edit script, or a diagnostic patch view.

`action_text`: Natural-language action used for headline inference.

`action_abs`: Deterministic abstract edit script derived from AST/CST changes.

`action_patch`: Diff-derived action view. It is leaky and used only for
diagnostics or upper bounds.

`transition`: One `(state_before, action, state_after)` training example.

`transition_energy`: Squared distance between predicted after-state latent and
candidate after-state latent.

`hard negative`: A wrong after-state chosen to be lexically or structurally
similar to the true after-state.

`collapse`: A failure mode where embeddings lose rank, variance, or neighborhood
diversity.

`SIGReg`: Sketch Isotropic Gaussian Regularizer used to prevent latent collapse.

`manifest`: Schema-versioned JSON file that records artifact lineage, config,
checksums, and parent artifacts.

`headline inference`: The user-facing path used for claims. It uses
`action_text`, not `action_patch`.

`patch surprise`: Evaluation that measures whether true or passing after-states
have lower transition energy than decoy after-states.
