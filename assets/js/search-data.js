// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-how-to-scale-your-model-如何扩展你的模型",
    title: "How to Scale Your Model（如何扩展你的模型）",
    section: "Navigation",
    handler: () => {
      window.location.href = "/scaling-book-zh/";
    },
  },{id: "dropdown-part-0-introduction",
              title: "Part 0. Introduction",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-1-intro-to-rooflines",
              title: "Part 1. Intro to Rooflines",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-2-all-about-tpus",
              title: "Part 2. All About TPUs",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-3-sharded-matmuls",
              title: "Part 3. Sharded Matmuls",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-4-transformers",
              title: "Part 4. Transformers",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-5-training",
              title: "Part 5. Training",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-6-training-llama",
              title: "Part 6. Training LLaMA",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-7-inference",
              title: "Part 7. Inference",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-8-serving-llama",
              title: "Part 8. Serving LLaMA",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-9-profiling",
              title: "Part 9. Profiling",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-10-all-about-jax",
              title: "Part 10. All About JAX",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-11-conclusions",
              title: "Part 11. Conclusions",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-12-gpus",
              title: "Part 12. GPUs",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-0-introduction",
              title: "Part 0. Introduction",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-1-intro-to-rooflines",
              title: "Part 1. Intro to Rooflines",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-2-all-about-tpus",
              title: "Part 2. All About TPUs",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-3-sharded-matmuls",
              title: "Part 3. Sharded Matmuls",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-4-transformers",
              title: "Part 4. Transformers",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-5-training",
              title: "Part 5. Training",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-6-training-llama",
              title: "Part 6. Training LLaMA",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-7-inference",
              title: "Part 7. Inference",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-8-serving-llama",
              title: "Part 8. Serving LLaMA",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-9-profiling",
              title: "Part 9. Profiling",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-10-all-about-jax",
              title: "Part 10. All About JAX",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-11-conclusions",
              title: "Part 11. Conclusions",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-part-12-gpus",
              title: "Part 12. GPUs",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-0-部分-介绍",
              title: "第 0 部分. 介绍",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-1-部分-关于屋顶线分析",
              title: "第 1 部分. 关于屋顶线分析",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-2-部分-如何理解-tpu",
              title: "第 2 部分. 如何理解 TPU",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-3-部分-分片矩阵与分片矩阵乘法",
              title: "第 3 部分. 分片矩阵与分片矩阵乘法",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-4-部分-你需要了解的-transformer-数学",
              title: "第 4 部分. 你需要了解的 Transformer 数学",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-5-部分-如何对-transformer-进行训练并行化",
              title: "第 5 部分. 如何对 Transformer 进行训练并行化",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-6-部分-在-tpu-上训练-llama-3",
              title: "第 6 部分. 在 TPU 上训练 LLaMA 3",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-7-部分-transformer-推理全解",
              title: "第 7 部分. Transformer 推理全解",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-8-部分-在-tpu-上部署服务-llama-3-70b",
              title: "第 8 部分. 在 TPU 上部署服务 LLaMA 3-70B",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-9-部分-如何理解-tpu-性能剖析",
              title: "第 9 部分. 如何理解 TPU 性能剖析",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-10-部分-用-jax-编写-tpu-程序",
              title: "第 10 部分. 用 JAX 编写 TPU 程序",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-11-部分-总结与延伸阅读",
              title: "第 11 部分. 总结与延伸阅读",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-12-部分-如何理解-gpu",
              title: "第 12 部分. 如何理解 GPU",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
