# Atendimento e Suporte de TI com LLM + RAG

> Sistema inteligente baseado em LLM para consulta e análise de documentos de atendimento e suporte de Tecnologia da Informação.

---

## Sobre o Projeto

Este projeto foi desenvolvido como **Trabalho Final** da disciplina de **Engenharia de Software para IA** e contempla o desenvolvimento de um **Sistema Inteligente Baseado em LLM (Large Language Model) para Consulta e Análise de Documentos**.

O sistema utiliza a técnica de **RAG (Retrieval-Augmented Generation)** para permitir que usuários façam perguntas em linguagem natural e recebam respostas contextualizadas com base em documentos institucionais reais. Neste caso, foram utilizados **2 manuais fictícios de atendimento e suporte de TI** como base de conhecimento para carregamento e análise pela LLM.

### Problema Resolvido

Em instituições públicas e privadas, manuais de atendimento e suporte de TI são extensos e de difícil consulta manual. Este sistema permite que qualquer usuário — técnico ou não — faça perguntas em linguagem natural e receba respostas precisas, baseadas exclusivamente no conteúdo dos documentos oficiais, reduzindo o tempo de busca e melhorando a eficiência do suporte.

---

## Funcionalidades

-  **Consulta em linguagem natural** — o usuário faz perguntas como se estivesse conversando com um atendente humano
-  **Análise de documentos** — carregamento e indexação de PDFs, DOCX e TXT
-  **Respostas contextualizadas** — baseadas exclusivamente no conteúdo dos documentos carregados (sem alucinação)
-  **Histórico de conversa** — mantém o contexto de mensagens anteriores para respostas mais coerentes
-  **Interface web com Streamlit** — UI amigável para o usuário final
-  **Segurança da informação** — respostas limitadas ao escopo dos documentos institucionais

---

## Tecnologias Utilizadas

 **LLM** - Groq API (Llama 3.3 70B) - Modelo de linguagem para geração de respostas 
 **Framework** - LangChain 0.3.x - Orquestração da pipeline RAG 
 **Embeddings** - HuggingFace (all-mpnet-base-v2) - Vetorização semântica dos documentos 
 **Banco Vetorial** - ChromaDB - Armazenamento e recuperação de embeddings 
 **Document Loaders** - PyMuPDF / python-docx - Extração de texto de PDF, DOCX e TXT 
 **Interface** - Streamlit - Framework web para a interface do usuário 
 **Ambiente** - Google Colab → IDE local - Desenvolvimento inicial e migração 

---

##  Arquitetura do Sistema

O sistema segue a arquitetura clássica de **RAG (Retrieval-Augmented Generation)** dividida em duas fases principais:

###  Fase 1 — Indexação
###  Fase 2 — Recuperação e Geração

O Projeto foi trabalhado e desenvolvido pelos alunos: Marcelo Hiroaki Ito e Max Flávio Cabral.
