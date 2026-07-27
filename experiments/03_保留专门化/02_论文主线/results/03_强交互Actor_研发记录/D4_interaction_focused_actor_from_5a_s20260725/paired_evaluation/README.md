# Epoch-7 Oracle Pairing Diagnostic

Status: diagnostic only; not a standalone-Actor or Gate result.

Both runs replay the same 140 validation scenario IDs in the same order.

| metric | frozen 5A | epoch-7 oracle combination |
| --- | ---: | ---: |
| agent success | 586/700 (`0.837`) | 601/700 (`0.859`) |
| full success | 66/140 (`0.471`) | 77/140 (`0.550`) |

Paired full-success outcomes:

| outcome | scenarios |
| --- | ---: |
| both succeed | 48 |
| 5A only | 18 |
| epoch-7 oracle combination only | 29 |
| neither succeeds | 45 |

The candidate run activates the trained Actor through a privileged interaction
distance oracle. It therefore measures the old training configuration's oracle
combination, not the strong Actor running independently. It cannot be used to
claim deployable Gate performance or expert complementarity.

Files:

- `base5a_r1.log`: frozen 5A reference.
- `epoch007_r1.log`: epoch-7 oracle combination.
