# The FREUID Challenge 2026 - IJCAI/ECAI

## FREUID Development Toolkit

Pacote auxiliar desenvolvido para facilitar o desenvolvimento, organização e experimentação de soluções para o **FREUID Challenge 2026 (IJCAI-ECAI)**.

O objetivo principal deste projeto é fornecer uma estrutura reutilizável para criação de notebooks de competição, reduzindo código repetitivo e permitindo uma abordagem mais organizada para treinamento, validação e submissão de modelos de Machine Learning e Deep Learning.

O desafio FREUID aborda a detecção de fraudes em documentos de identidade, incluindo manipulações físicas, alterações geradas por modelos de IA e falsificações por captura/impressão, exigindo modelos robustos para generalização em diferentes cenários. :contentReference[oaicite:2]{index=2}

---

## Objetivos

Este pacote foi criado para auxiliar participantes durante o ciclo de desenvolvimento da solução:

- Organização do pipeline experimental
- Padronização do treinamento de modelos
- Facilitação de testes e validações
- Reutilização de componentes dentro de notebooks
- Simplificação do processo de submissão
- Separação entre lógica de competição e experimentação

---

## Principais funcionalidades

### Dataset Management

Ferramentas para:

- Carregamento dos dados
- Organização de caminhos
- Preparação de imagens
- Criação de estruturas compatíveis com frameworks de Deep Learning

---

### Training Workflow

Abstrações para facilitar:

- Treinamento de modelos
- Controle de etapas do pipeline
- Validação
- Monitoramento de experimentos

Exemplo de fluxo:
