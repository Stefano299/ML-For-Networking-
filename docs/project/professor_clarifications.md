# Clarifications & Technical Questions for the Professor / Tutor

> **Course:** Machine Learning for Networking (ML4N) — Group Project 4  
> **Topic:** IP Flow Classification, Feature Preprocessing & Domain Nuances

---

## 1. Handling `Source.Port` & Reversed Network Flows

### Problem Statement
In IP networking, the source port is typically **ephemeral** (a random high port chosen by the client, e.g., `51234`), while the destination port represents the well-known server service (e.g., `443` for HTTPS, `80` for HTTP).

However, due to packet capture initialization in `CICFlowMeter`, **~6% of total flows** (and **~64% of Dropbox flows**) have inverted client/server roles:
- **Normal Flow:** `Source.Port` = `51234` (client, random), `Destination.Port` = `443` (server, service)
- **Inverted Flow:** `Source.Port` = `443` (server, service), `Destination.Port` = `51234` (client, random)

In inverted flows, the informative server port is stored in `Source.Port`, while `Destination.Port` holds the ephemeral port (which gets categorized into `"Other"`).

### Proposed Options
1. **Option 1 (Baseline / Current implementation):**  
   Drop `Source.Port` completely. Simple and prevents the classifier from learning spurious noise from ephemeral ports, but in inverted flows the server port information is lost.
2. **Option 2 (Bidirectional Service Port Extraction):**  
   Extract a unified `service_port` attribute: select the known service port (`80`, `443`, `3128`, ...) regardless of whether it appears in `Source.Port` or `Destination.Port`. Add a binary flag `reversed_flow` (0 or 1) indicating whether the flow direction was flipped. Ephemeral port numbers are discarded in both cases.

### Question for the Professor / Tutor:
> *"The source port is ephemeral in ~94% of flows, but in ~6% (and 64% of Dropbox traffic) the flow capture is reversed and the service port ends up in `Source.Port`. Is dropping `Source.Port` completely acceptable for the project evaluation, or should we reconstruct a canonical `service_port` feature with an `is_reversed` indicator?"*

---

## 2. Sentinel Value `-1` in TCP Window Fields

### Problem Statement
In fields like `Init_Win_bytes_forward` and `Init_Win_bytes_backward`, `CICFlowMeter` records `-1` whenever the TCP window size could not be determined (e.g., missing initial SYN packet or non-TCP flows).

### Question for the Professor / Tutor:
> *"How should sentinel values of `-1` in `Init_Win_bytes_*` be treated during feature scaling and normalization? Should `-1` be replaced with `0`, treated as missing data (imputed with median/mean), or flagged with a separate binary indicator?"*

---

## 3. Min-Max Normalization Interval (PDF Specification 1.2)

### Problem Statement
In Section 1.2 of the Project 4 specification PDF, the text reads:  
> *"map all features in the interval ."* (the interval values are missing from the typography).

### Clarification:
> *Can you confirm that the expected normalization range is standard $[0, 1]$ via `MinMaxScaler` (fitted strictly on the training set)?*
