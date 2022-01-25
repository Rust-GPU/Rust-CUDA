// use quote::quote;
// use syn::*;

// #[proc_macro_attribute]
// pub fn launch_params(_input: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
//     let statik = parse_macro_input!(item as ItemStatic);
//     let ty = statik.ty;
//     let name = statik.ident;
//     let vis = statik.vis;
//     let tokens = quote! {
//         extern "C" {

//         }
//     }
// }
