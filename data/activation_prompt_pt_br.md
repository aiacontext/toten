INSTRUÇÃO: O texto a seguir contém anotações estruturais entre colchetes que constituem a representação canônica do framework TOTEN. Trate-as como contrato:

- `[QTY value=N unit=U dim=[...]]` — grandeza física com valor, unidade e vetor dimensional ℤ⁷ canônico na ordem `[kg, m, s, A, K, mol, cd]`.
- `[QTY ... unc=N]` — grandeza com incerteza na mesma unidade.
- `[IDX:slug]` — identificador técnico ATÔMICO (norma, material, símbolo). Entidade indivisível; nunca fragmente.
- `[CONST:simbolo value=N unit=U]` — constante universal com valor canônico já fornecido; não substitua por aproximação.
- `[SYM:<canonical>]` — expressão simbólica em variáveis livres (ex.: `[SYM:p*l/12]`, `[SYM:p*l**2*2**(1/3)/32]`). Conteúdo é a forma canônica SymPy ASCII reparseável. Entidade ATÔMICA — não fragmente; preserve a estrutura algébrica intacta nas respostas.
- `[REF:<hierarquia>]` — referência hierárquica a parágrafo, artigo, inciso, alínea, seção (ex.: `[REF:8.2.1]`, `[REF:14.2.3]`). Entidade ATÔMICA referenciando estrutura legal/normativa; não interprete como valor numérico nem aplique aritmética. Sempre vem precedida de enumerador normativo (`§`, `parágrafo`, `art.`, `inciso`, etc.) preservado como prosa.

Regras de uso obrigatórias:

1. **Fonte definitiva.** Os valores numéricos, unidades, identidades e dimensões dessas anotações são fonte de verdade. Não substitua, não arredonde prematuramente, não aproxime por memória.
2. **Homogeneidade dimensional.** O vetor `dim` codifica `[mass, length, time, current, temperature, amount, luminosity]` no SI. Use para verificar consistência entre grandezas (somar requer dims idênticos; multiplicar/dividir compõe dims).
3. **Distinção de identificadores.** Identificadores `[IDX:...]` são entidades distintas com propriedades específicas. Não confunda variantes próximas — por exemplo, `[IDX:sae-4140]` e `[IDX:sae-4140-h]` têm propriedades diferentes (H = high hardenability); `[IDX:aisi-304]` e `[IDX:aisi-304-l]` também (L = low carbon).
4. **Preservação da convenção do engenheiro.** Nas respostas, mantenha a unidade que o engenheiro usou (MPa, kgf/cm², tf, etc.). Só converta para SI se explicitamente solicitado.
5. **Grounding contra confabulação.** Quando uma constante aparece como `[CONST:simbolo value=N unit=U]`, use exatamente o valor fornecido. Quando uma norma aparece como `[IDX:nbr-XXXX]`, cite a cláusula apenas se você tem certeza factual; caso contrário, declare explicitamente a incerteza.
6. **Expressões simbólicas.** Quando `[SYM:<canonical>]` aparece, é entidade atômica — uma composição mediada de variáveis e operadores. A resposta canônica preserva a forma fechada — não substitua variáveis livres por valores nem avalie numericamente sem instrução explícita. Quando o problema é paramétrico (resposta em `p`, `l`, etc.), reporte expressões equivalentes (`p*l/12` ≡ `(1/12)*p*l`) mas mantenha a forma simbólica.

7. **Marcadores de ambiguidade declarada.** Quando uma tag carrega `ambig="<tipo>"`, NÃO assuma a interpretação primária — o tokenizer detectou estruturalmente uma ambiguidade. O campo `alternatives="<a1>|<a2>|..."` lista as interpretações possíveis no formato `<chave>:<descrição_curta>`.

   Tipos catalogados (ver SPEC_07):
   - `unit-letter`: letra ASCII maiúscula ambígua entre unidade SI (A=ampere, V=volt, K=kelvin, W=watt, N=newton, J=joule, T=tesla, H=henry, F=farad, C=coulomb, S=siemens) e letra de seção/referência.
   - `domain-semantic`: significado dependente de domínio (ex.: `N` em física = newton; em química = normal).
   - `typography-degraded`: tipografia sugere expoente mas é literal (ex.: `m s−2` deveria ser `m·s⁻²`).
   - `locale-thousands`: separador numérico ambíguo entre PT-BR (`1.500`=1500) e EN (`1.500`=1.5).

   Antes de prosseguir, faça UMA pergunta de confirmação ao usuário citando o trecho original e as alternativas em linguagem natural:

   > *"No trecho '<contexto>', '<original>' pode ser interpretado como:*
   > *1. <alternativa_1_descrita>*
   > *2. <alternativa_2_descrita>*
   > *A qual você se refere?"*

   Não invente interpretação. Não escolha silenciosamente. Se o contexto da conversa anterior já resolveu a ambiguidade (ex.: o usuário declarou "estamos em química"), use essa resolução sem perguntar novamente.

Tarefa:
